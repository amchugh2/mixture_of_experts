from experts import MarketDataExpert
import pandas as pd

df = pd.read_csv('VOO-20200628-20250627.ohlcv-1d.csv')

# Create an instance of MarketDataExpert and analyze the data
market_expert = MarketDataExpert()
result = market_expert.analyze(df)

print(result)