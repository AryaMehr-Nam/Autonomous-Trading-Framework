import MetaTrader5 as mt5

if not mt5.initialize():
    print("MT5 init failed")
    exit()

symbols = mt5.symbols_get()
names = [s.name for s in symbols] if symbols else []
print(f"Total symbols: {len(names)}")
matches = [n for n in names if any(k in n.upper() for k in ["30", "DOW", "DJ", "WALL", "US30", "XAU"])]
print(f"Matches found: {matches[:30]}")

mt5.shutdown()
