import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def fetch_data(tickers, start="2020-01-01", end=None):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    return data["Close"]

# Ticker Lookup
tickers_input = input("Enter ticker symbols (comma‑separated), e.g. AAPL,MSFT,GOOG: ")
cleaned_tickers = []
for t in tickers_input.split(","):
    cleaned = t.strip().upper()
    cleaned_tickers.append(cleaned)
tickers = cleaned_tickers

# Fetch data for the user’s tickers
adj_close = fetch_data(tickers)
if adj_close.empty:
    print("No data was downloaded. You may be rate-limited.")
    exit()
adj_close.to_csv("data/adj_close.csv")


# Core metrics
daily_returns      = adj_close.pct_change().dropna()
cumulative_returns = (1 + daily_returns).cumprod()
volatility         = daily_returns.std() * np.sqrt(252)
correlation        = daily_returns.corr()

# Naive Linear‐Trend Forecast
horizon = int(input("\nEnter forecast horizon in trading days (e.g. 30): "))

for ticker in tickers:
    prices = adj_close[ticker].values
    days = np.arange(len(prices))
    slope, intercept = np.polyfit(days, prices, 1)
    future_day = len(prices) + horizon
    forecast = slope * future_day + intercept
    print("\n", ticker, "forecast (", horizon, "days ahead): $", round(forecast, 2))


    # Plot last 60 days + forecast point
    recent = adj_close[ticker].tail(60)
    plt.figure(figsize=(8,4))
    plt.plot(recent.index, recent.values, label=f"{ticker} Price")
    plt.scatter(recent.index[-1] + pd.Timedelta(days=horizon),
                forecast, color="red", label="Forecast")
    plt.title(f"{ticker}: Last 60 Days + {horizon}‑Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Display results
print(f"\nMetrics for: {', '.join(tickers)}")
print("\nDaily returns (most recent 5 rows):")
print(daily_returns.tail())
print("\nAnnualized volatility:")
print(volatility)
print("\nCorrelation matrix:")
print(correlation)

# Portfolio Allocation Recommendation
portfolio_value = float(input("\nEnter total portfolio value in USD (e.g. 10000): "))

# Calculate inverse‑volatility weights
inv_vol = 1 / volatility
weights_iv = inv_vol / inv_vol.sum()

# Calculate dollar allocations
allocations = weights_iv * portfolio_value

print("\nRecommended allocation (inverse-volatility weighting):")
for ticker, w in weights_iv.items():
    print(f"  {ticker}: {w:.2%} → ${allocations[ticker]:,.2f}")


