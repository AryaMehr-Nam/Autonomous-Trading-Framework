import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import Order, OrderType, OrderSide, AssetClass
from execution.brokers.mt5_executor import MT5Executor

async def main():
    executor = MT5Executor()
    connected = await executor.connect()
    if not connected:
        print("Failed to connect to MT5")
        return
        
    order = Order(
        symbol="XAU/USD",
        asset_class=AssetClass.FOREX,
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=0.01,
        stop_loss=0.0,
        take_profit=0.0
    )
    
    print("Sending Buy Market Order for XAU/USD (0.01 lots)...")
    res = await executor.open_order(order)
    print(f"Result: {res.status.value if hasattr(res.status, 'value') else res.status}, ID: {res.broker_order_id}")
    await executor.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
