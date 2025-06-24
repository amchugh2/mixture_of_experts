#! /usr/bin/python
from __future__ import print_function
import sys
sys.path.append('..')
from orderbook import OrderBook
import pandas as pd

class RealOrderFeeder:
    def __init__(self, mbo_file_path, symbol="VOO"):
        print("[INIT] Loading MBO file:", mbo_file_path)
        self.df = pd.read_csv(mbo_file_path, parse_dates=["ts_event"])
        print(f"[INIT] Loaded {len(self.df)} rows total")

        self.df = self.df[(self.df["symbol"] == symbol) & (self.df["action"] == "A")]
        print(f"[INIT] Filtered to {len(self.df)} 'Add' orders for symbol '{symbol}'")

        self.df.sort_values("ts_event", inplace=True)
        self.orders = self.df.reset_index(drop=True)
        self.pointer = 0
        print("[INIT] RealOrderFeeder ready\n")

    def get_next_order(self):
        if self.pointer >= len(self.orders):
            print("[Feeder] No more orders to feed")
            return None

        row = self.orders.iloc[self.pointer]
        self.pointer += 1
        order = {
            "type": "limit",
            "side": "bid" if row["side"] == "B" else "ask",
            "quantity": int(row["size"]),
            "price": float(row["price"]),
            "trade_id": int(row["order_id"])
        }
        print(f"[Feeder] Feeding order {self.pointer}/{len(self.orders)}: {order}")
        return order

def stream_orders_into_book(order_book, order_feeder, verbose=False):
    print("[Streamer] Starting to stream orders into order book")
    count = 0
    while True:
        order = order_feeder.get_next_order()
        if order is None:
            break
        order_book.process_order(order, False, verbose)
        count += 1
    print(f"[Streamer] Finished streaming {count} orders into the book")
