from orderbook import OrderBook
from agent_traders import MeanRevertingTrader, TrendFollowingTrader, NoiseTrader
import pandas as pd
import time

# === Step 1: Load real VOO prices from MBO ===
mbo_file = "xnas-itch-20250516.mbo.csv"
mbo_df = pd.read_csv(mbo_file, parse_dates=["ts_event"])

# Filter for VOO add orders
voo_adds = mbo_df[(mbo_df["symbol"] == "VOO") & (mbo_df["action"] == "A")]
voo_adds = voo_adds.sort_values("ts_event")

# Separate bids and asks
bids = voo_adds[voo_adds["side"] == "B"].copy()
asks = voo_adds[voo_adds["side"] == "A"].copy()

# Print raw counts for debugging
print(f"Total bids: {len(bids)}")
print(f"Total asks: {len(asks)}")

# === Round timestamps to 1-second bins ===
bids["ts_bin"] = bids["ts_event"].dt.floor("1s")
asks["ts_bin"] = asks["ts_event"].dt.floor("1s")

# === Group by bin and compute best prices ===
bid_prices = bids.groupby("ts_bin")["price"].max()
ask_prices = asks.groupby("ts_bin")["price"].min()

# Merge and compute mid-prices
mid_prices = pd.concat([bid_prices, ask_prices], axis=1).dropna()
mid_prices.columns = ["bid", "ask"]
mid_prices["mid"] = (mid_prices["bid"] + mid_prices["ask"]) / 2

# Create list of mid-prices for traders
prices = mid_prices["mid"].tolist()
print(f"Loaded {len(prices)} mid-prices from 1-second bins.")

# === Step 2: Initialize the OrderBook and Traders ===
order_book = OrderBook()
traders = [
    MeanRevertingTrader(trader_id=1, initial_price=prices[0]),
    TrendFollowingTrader(trader_id=2, initial_price=prices[0]),
    NoiseTrader(trader_id=3, initial_price=prices[0])
]

# List to log all trades
all_trades_log = []

def print_order_book_snapshot(order_book, levels=10):
    print("------ Order Book Snapshot ------")

    sorted_bids = sorted(order_book.bids.price_map.items())[:levels]
    sorted_asks = sorted(order_book.asks.price_map.items())[:levels]

    print("Top Bids:")
    for price, orders in sorted_bids:
        total_qty = sum(order.quantity for order in orders)
        print(f"  {price:.2f} | {total_qty} shares")

    print("Top Asks:")
    for price, orders in sorted_asks:
        total_qty = sum(order.quantity for order in orders)
        print(f"  {price:.2f} | {total_qty} shares")

    if sorted_bids:
        print(f">>> Best Bid: {sorted_bids[0][0]:.2f}")
    if sorted_asks:
        print(f">>> Best Ask: {sorted_asks[0][0]:.2f}")
    print("---------------------------------\n")

# === Step 3: Feed strategy-driven orders into the book ===
# === Step 3: Feed strategy-driven orders into the book ===
for t in range(1, len(prices)):
    for trader in traders:
        order = trader.get_order(t, prices)
        if order:
            trader_type = trader.__class__.__name__
            trader_id = trader.trader_id

            if order['price'] > 800 or order['price'] < 100:
                print(f"[OUTLIER DETECTED] {trader_type} (Trader {trader_id}) placed {order['side'].upper()} at {order['price']:.2f}")

            print(f"\n=== TIME {t} ===")
            print(f"[BEFORE ORDER] {trader_type} (Trader {trader_id}) placing {order['side'].upper()} for {order['quantity']} @ {order['price']:.2f}")
            print_order_book_snapshot(order_book)

            trades, neworder = order_book.process_order(order, False, False)

            if trades:
                for trade in trades:
                    print(f"[MATCH] {trader_type} (Trader {trader_id}) filled {trade['quantity']} @ {trade['price']:.2f}")
                    all_trades_log.append({
                        'time_index': t,
                        'trader_id': trader_id,
                        'trader_type': trader_type,
                        'side': order['side'],
                        'quantity': trade['quantity'],
                        'price': trade['price']
                    })

            if not trades and neworder:
                print(f"[UNFILLED] {trader_type} (Trader {trader_id}) {order['side'].upper()} unfilled at {order['price']:.2f}")

            print(f"[AFTER ORDER] Book state after processing {trader_type}'s order:")
            print_order_book_snapshot(order_book)
            time.sleep(0.15)  

# === Final state ===
#print("\nFinal order book state:")
#print(order_book)

# === Print all logged trades ===
#print("\n=== Full Trade Log ===")
#for trade in all_trades_log:
#    print(trade)
