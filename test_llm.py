import asyncio
import pandas as pd
from core.models import Signal, SignalType, Timeframe
from signals.context_agent import MarketContext
from signals.llm_validator import LLMTradeValidator

async def test_validator():
    print("Testing LLMTradeValidator...")
    validator = LLMTradeValidator()
    
    # We force enable it for testing in case config parsing failed
    validator.enabled = True
    
    # Create fake Signal
    sig = Signal(
        id="test-123",
        symbol="XAU/USD",
        timeframe=Timeframe.H1,
        signal_type=SignalType.BUY,
        limit_price=2400.0,
        stop_loss=2390.0,
        take_profit=2420.0,
        confidence=0.85
    )
    
    # Create fake DataFrame row (last candle)
    data = {
        'close': [2405.0],
        'RSI': [45.0],
        'MACD_LINE': [1.5],
        'MACD_HIST': [0.2],
        'EMA_50': [2395.0],
        'EMA_200': [2350.0],
        'RVOL': [1.5]
    }
    df = pd.DataFrame(data)
    
    ctx = MarketContext.STRONG_BULLISH
    
    print("Sending payload to Kimi API (TokenRouter)...")
    result = await validator.validate_signal(sig, df, ctx)
    print(f"\nFinal Result: {result}")

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_validator())
