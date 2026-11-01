import asyncio
from execution.brokers.mt5_executor import MT5Executor

async def check():
    executor = MT5Executor()
    connected = await executor.connect()
    if not connected:
        print("Failed to connect to MT5.")
        return
        
    info = await executor.get_account_info()
    print(f"STARTING_BALANCE: {info.balance}")
    print(f"STARTING_EQUITY: {info.equity}")
    print(f"UNREALIZED_PNL: {info.unrealized_pnl}")
    
    await executor.disconnect()

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check())
