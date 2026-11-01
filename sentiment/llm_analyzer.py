"""
TradeAgent LLM Analyzer.

Uses LLM (OpenAI) to perform deep analysis of complex macroeconomic events,
such as FOMC statements or central bank speeches.
"""

import asyncio
import json
from typing import Dict, Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from config.settings import settings
from core.logger import get_logger
from core.models import SentimentLabel, SentimentResult

logger = get_logger("llm_analyzer")


class LLMAnalyzer:
    def __init__(self):
        self.api_key = settings.openai_api_key

    def is_configured(self) -> bool:
        return HAS_HTTPX and bool(self.api_key)

    async def analyze_macro_statement(self, text: str) -> Optional[Dict]:
        """
        Analyze a central bank statement using OpenAI GPT.
        Returns a structured dictionary with hawkish/dovish score.
        """
        if not self.is_configured():
            logger.warning("LLM Analyzer not configured (missing API key or httpx).")
            return None
            
        prompt = f"""
        Analyze the following macroeconomic statement from a central bank.
        Classify the monetary policy stance as 'Hawkish', 'Dovish', or 'Neutral'.
        Provide an impact_score from -1.0 (very Dovish) to +1.0 (very Hawkish).
        Identify the currency pairs most likely to be affected.
        
        Return ONLY valid JSON matching this schema:
        {{
            "sentiment": "Hawkish" | "Dovish" | "Neutral",
            "impact_score": float,
            "affected_pairs": ["EUR/USD", "USD/JPY"],
            "reasoning": "Brief 1-sentence explanation"
        }}
        
        Statement:
        {text[:2000]} # Truncate to save tokens
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                result = json.loads(content)
                logger.info(f"LLM Macro Analysis: {result.get('sentiment')} ({result.get('impact_score')})")
                return result
            else:
                logger.error(f"OpenAI API Error: {res.text}")
                return None
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None
