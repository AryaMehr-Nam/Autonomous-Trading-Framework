import asyncio
import sys
from core.models import Order, OrderSide, OrderType, Timeframe, AssetClass
from execution.brokers.mt5_executor import MT5Executor
from data.providers.mt5_provider import MT5DataProvider

async def test_order_placement():
    print("Testing MT5 Live Order Placement Pipeline...")
    provider = MT5DataProvider()
    connected = await provider.connect()
    if not connected:
        print("Failed to connect to MT5 Provider")
        return
        
    executor = MT5Executor()
    exec_conn = await executor.connect()
    if not exec_conn:
        print("Failed to connect to MT5 Executor")
        return
        
    acc = await executor.get_account_info()
    print(f"MT5 Account Connected: Balance={acc.balance} USD, Equity={acc.equity} USD")
    
    positions = await executor.get_open_positions()
    print(f"Current Open Positions: {len(positions)}")
    
    df = await provider.get_historical_dataframe("EUR/USD", Timeframe.M5, limit=10)
    if df is not None and not df.empty:
        curr_price = df.iloc[-1]['close']
        print(f"EUR/USD Current Price: {curr_price}")
        
        # Test 0.01 lot order with precise SL and TP
        test_order = Order(
            symbol="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            price=curr_price,
            stop_loss=round(curr_price - 0.0008, 5), # 8 pips SL
            take_profit=round(curr_price + 0.0016, 5), # 16 pips TP (1:2 RR)
        )
        print("Placing 0.01 lot BUY order on EUR/USD in MT5...")
        res = await executor.open_order(test_order)
        print(f"Order Result Status: {res.status}")
        
        open_pos = await executor.get_open_positions()
        print(f"Total Open Positions after placement: {len(open_pos)}")
        for p in open_pos:
            print(f"   -> Live Trade: Ticket={p.id}, Pair={p.symbol}, Side={p.side}, Open={p.entry_price}, SL={p.stop_loss}, TP={p.take_profit}")
            
    await provider.disconnect()
    await executor.disconnect()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_order_placement())
