"""
TradeAgent - Main Entrypoint.

Starts the autonomous trading agent.
"""

import asyncio
import sys
import argparse
from dotenv import load_dotenv

# Load .env file before anything else
load_dotenv(dotenv_path="config/.env")

from config.settings import settings
from core.logger import init_logging, get_logger
from core.agent import TradeAgent


async def main():
    parser = argparse.ArgumentParser(description="TradeAgent Autonomous Trading Bot")
    parser.add_argument("--mode", type=str, choices=["paper", "live"], help="Override trade mode (default from .env)")
    parser.add_argument("--log-level", type=str, help="Override log level")
    
    args = parser.parse_args()
    
    if args.mode:
        settings.trade_mode = args.mode
    if args.log_level:
        settings.log_level = args.log_level
        
    init_logging(settings.log_level)
    logger = get_logger("main")
    
    logger.info("=" * 60)
    logger.info("🤖 Starting TradeAgent")
    logger.info(f"Mode:   {settings.trade_mode.upper()}")
    logger.info(f"Market: {settings.default_market.upper()}")
    logger.info("=" * 60)
    
    agent = TradeAgent()
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        await agent.stop()
        logger.info("TradeAgent shut down gracefully.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
