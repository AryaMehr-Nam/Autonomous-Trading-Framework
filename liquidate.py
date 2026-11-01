import asyncio
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")
from execution.brokers.mt5_executor import MT5Executor

async def liquidate_all():
    exec = MT5Executor()
    connected = await exec.connect()
    if not connected:
        print("Failed to connect to MT5.")
        return
        
    positions = await exec.get_open_positions()
    print(f"Found {len(positions)} open positions. Liquidating...")
    
    for p in positions:
        print(f"Closing [{p.symbol}] {p.side.name}...")
        # MT5 close position needs current price, but for market orders we can just pass 0.0
        success = await exec.close_position(p, 0.0)
        if success:
            print(f"Successfully closed {p.symbol}")
        else:
            print(f"Failed to close {p.symbol}")

    await exec.disconnect()
    print("Liquidation complete.")

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(liquidate_all())
