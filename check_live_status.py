import MetaTrader5 as mt5
import sys

def check_status():
    if not mt5.initialize():
        print("MT5 initialize failed")
        sys.exit(1)

    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info")
        mt5.shutdown()
        sys.exit(1)

    positions = mt5.positions_get()
    orders = mt5.orders_get()

    print("=== LIVE ACCOUNT STATUS ===")
    print(f"Balance: {account_info.balance}")
    print(f"Equity:  {account_info.equity}")
    print(f"Margin:  {account_info.margin}")
    print(f"Free Margin: {account_info.margin_free}")
    print(f"Margin Level: {account_info.margin_level}%")
    
    print("\n=== OPEN POSITIONS ===")
    if positions is None or len(positions) == 0:
        print("No open positions.")
    else:
        for pos in positions:
            print(f"[{pos.ticket}] {pos.symbol} {'BUY' if pos.type == 0 else 'SELL'} | Vol: {pos.volume} | Open: {pos.price_open} | Current: {pos.price_current} | Profit: {pos.profit}")
            
    print("\n=== PENDING ORDERS ===")
    if orders is None or len(orders) == 0:
        print("No pending orders.")
    else:
        for ord in orders:
            print(f"[{ord.ticket}] {ord.symbol} | Vol: {ord.volume_initial} | State: {ord.state}")

    mt5.shutdown()

if __name__ == "__main__":
    check_status()
