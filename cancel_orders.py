import MetaTrader5 as mt5
import sys

def cancel_all_pending():
    if not mt5.initialize():
        print("MT5 initialize failed")
        sys.exit(1)

    orders = mt5.orders_get()
    if orders is None or len(orders) == 0:
        print("No pending orders to cancel.")
        mt5.shutdown()
        return

    print(f"Found {len(orders)} pending orders. Canceling...")
    
    for ord in orders:
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ord.ticket,
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Canceled order {ord.ticket}")
        else:
            print(f"Failed to cancel {ord.ticket}: {res.comment}")

    mt5.shutdown()

if __name__ == "__main__":
    cancel_all_pending()
