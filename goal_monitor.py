import asyncio
import time
from execution.brokers.mt5_executor import MT5Executor
from core.logger import get_logger

logger = get_logger("goal_monitor")

async def monitor():
    executor = MT5Executor()
    await executor.connect()
    
    start_time = time.time()
    target_profit = 50.0
    
    info = await executor.get_account_info()
    start_equity = info.equity
    logger.info(f"Goal Monitor Started. Starting Equity: {start_equity}. Target: {start_equity + target_profit}")
    
    while True:
        info = await executor.get_account_info()
        profit = info.equity - start_equity
        logger.info(f"Current Profit: ${profit:.2f} | Time Elapsed: {(time.time() - start_time)/3600:.2f} hours")
        
        if profit >= target_profit:
            logger.info("🎯 GOAL REACHED! $50 Profit achieved.")
            break
            
        elapsed = time.time() - start_time
        if elapsed > 5 * 3600:
            logger.info("⏱️ TIME UP! 5 hours elapsed.")
            break
            
        await asyncio.sleep(300) # check every 5 minutes
        
    await executor.disconnect()

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(monitor())
