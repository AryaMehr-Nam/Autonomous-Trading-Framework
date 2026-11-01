import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed")
    sys.exit(1)

async def test_connect():
    login = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")
    path = os.getenv("MT5_PATH", "")
    
    print(f"Login: {login}")
    print(f"Server: {server}")
    print(f"Path: {path}")
    print(f"Password: {'*' * len(password)}")
    
    # Try with path first
    if path:
        ok = mt5.initialize(path=path, login=login, server=server, password=password)
        if ok:
            print("Connected WITH path!")
        else:
            err = mt5.last_error()
            print(f"Failed with path: {err}")
            # Try without path
            mt5.shutdown()
            ok = mt5.initialize(login=login, server=server, password=password)
            if ok:
                print("Connected WITHOUT path!")
            else:
                err = mt5.last_error()
                print(f"Failed without path too: {err}")
                
                # Try bare init (connect to whatever MT5 is already running)
                mt5.shutdown()
                ok = mt5.initialize()
                if ok:
                    print("Connected with BARE init (existing terminal)!")
                else:
                    err = mt5.last_error()
                    print(f"All methods failed: {err}")
                    return
    else:
        ok = mt5.initialize(login=login, server=server, password=password)
        if not ok:
            print(f"Failed: {mt5.last_error()}")
            return
    
    acc = mt5.account_info()
    if acc:
        print(f"\nAccount: {acc.login}")
        print(f"Balance: {acc.balance} {acc.currency}")
        print(f"Equity: {acc.equity} {acc.currency}")
        print(f"Server: {acc.server}")
    else:
        print("Connected but could not fetch account info")
    
    mt5.shutdown()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_connect())
