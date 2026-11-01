import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")

mt5.initialize(path=os.getenv("MT5_PATH"))
login = int(os.getenv("MT5_LOGIN", 0))
password = os.getenv("MT5_PASSWORD", "")
server = os.getenv("MT5_SERVER", "MetaQuotes-Demo")

mt5.login(login=login, password=password, server=server)

symbols = mt5.symbols_get()
crypto_syms = [s.name for s in symbols if "BTC" in s.name or "ETH" in s.name or "CRYPTO" in s.path.upper()]
forex_syms = [s.name for s in symbols if "EURUSD" in s.name or "GBPUSD" in s.name]

print(f"Total MT5 Symbols Available: {len(symbols)}")
print(f"Forex Symbols Found: {forex_syms}")
print(f"Crypto (24/7 Weekend) Symbols Found: {crypto_syms}")

# Check trade mode / trading status for BTC symbols
for sym in crypto_syms[:5]:
    info = mt5.symbol_info(sym)
    if info:
        print(f"  -> {sym}: Visible={info.visible}, TradeMode={info.trade_mode}, Bid={info.bid}, Ask={info.ask}")

mt5.shutdown()
