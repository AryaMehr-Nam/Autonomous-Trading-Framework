import json
import os
import asyncio
from core.logger import get_logger
from core.models import Signal
from signals.context_agent import MarketContext
import pandas as pd

logger = get_logger("llm_validator")

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class LLMTradeValidator:
    def __init__(self):
        self.client = None
        self.enabled = False
        self.setups_prompt = ""
        
        self._load_config()
        
    def _load_config(self):
        if not HAS_OPENAI:
            logger.warning("openai package not installed. LLMTradeValidator disabled.")
            return
            
        try:
            from config.settings import trading_params
            llm_cfg = trading_params.ml.get("llm_validation", {})
            self.enabled = llm_cfg.get("enabled", False)
            self.mode = llm_cfg.get("mode", "strict")
        except Exception:
            self.enabled = False
            
        if self.enabled:
            # Using TokenRouter skill settings
            self.client = AsyncOpenAI(
                base_url=os.getenv('TOKENROUTER_BASE_URL', 'https://api.tokenrouter.com/v1'),
                api_key=os.getenv('TOKENROUTER_API_KEY', ''),
            )
            # Load the setups prompt
            setup_path = os.path.join(os.path.dirname(__file__), "..", "config", "setups.md")
            try:
                with open(setup_path, "r", encoding="utf-8") as f:
                    self.setups_prompt = f.read()
            except Exception as e:
                logger.error(f"Failed to read setups.md: {e}")
                self.setups_prompt = "You are a strict trading assistant. Evaluate the JSON payload."

    async def validate_signal(self, signal: Signal, df: pd.DataFrame, context: MarketContext, macro_bias: str = "MACRO_NEUTRAL", h1_summary: dict = None) -> str:
        """
        Calls TokenRouter API (moonshotai/kimi-k3-free) to validate the trade setup.
        Returns "confirmed", "watch", or "invalid".
        """
        if not self.enabled or not self.client:
            return "confirmed" # Pass through if disabled
            
        if df.empty:
            return "watch"
            
        latest = df.iloc[-1]
        
        # Build JSON Payload for the LLM
        payload = {
            "setup": {
                "asset": signal.symbol,
                "direction": signal.signal_type.name if hasattr(signal.signal_type, 'name') else str(signal.signal_type),
                "current_price": float(latest.get('close', 0.0)),
                "timeframe": signal.timeframe.name if hasattr(signal.timeframe, 'name') else str(signal.timeframe)
            },
            "technical_indicators": {
                "RSI": float(latest.get('RSI', 50.0)),
                "MACD_LINE": float(latest.get('MACD_LINE', 0.0)),
                "MACD_HIST": float(latest.get('MACD_HIST', 0.0)),
                "EMA_50": float(latest.get('EMA_50', 0.0)),
                "EMA_200": float(latest.get('EMA_200', 0.0)),
                "RVOL": float(latest.get('RVOL', 1.0))
            },
            "market_context": {
                "micro_regime": context.name if hasattr(context, 'name') else str(context),
                "macro_h1_bias": macro_bias,
                "macro_h1_details": h1_summary if h1_summary else {}
            }
        }
        
        messages = [
            {"role": "system", "content": self.setups_prompt},
            {"role": "user", "content": json.dumps(payload, indent=2)}
        ]
        
        try:
            # We use await client.chat.completions.create for async
            response = await self.client.chat.completions.create(
                model="moonshotai/kimi-k3-free",
                messages=messages,
                temperature=0.0 # Keep it strictly deterministic
            )
            
            content = response.choices[0].message.content
            # Try to parse JSON. Sometimes LLMs return markdown blocks.
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            try:
                result = json.loads(content)
                status = result.get("status", "watch").lower()
                reason = result.get("reason", "No reason provided")
                logger.info(f"[LLM VALIDATION] {signal.symbol} {signal.signal_type} -> {status.upper()}: {reason}")
                return status
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM: {content}")
                return "watch" # Fallback safely
                
        except Exception as e:
            logger.error(f"Error calling TokenRouter LLM: {e}")
            return "watch" # Fallback safely if API fails
