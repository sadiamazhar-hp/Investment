# from alpha_vantage.timeseries import TimeSeries
# ts = TimeSeries(key="N6A6QT6IBFJOPJ70", output_format="pandas")
# data, meta = ts.get_intraday(symbol="AAPL", interval="5min")
# print(data.head())
import yfinance as yf
# df = yf.download("AAPL", start="2024-01-01", end="2025-03-13")
# print(df.head())
df = yf.download("MSFT", start="2024-01-01", end="2025-03-13")
print(df.head())
