import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def fetch_data(tickers, start="2020-01-01", end=None):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    return data["Close"]

#Ticker Lookup
tickers_input = input("Enter ticker symbols (comma‑separated), e.g. AAPL,MSFT,GOOG: ")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

#Fetch data for the user’s tickers
adj_close = fetch_data(tickers)
adj_close.to_csv("data/adj_close.csv")

#Core metrics
daily_returns      = adj_close.pct_change().dropna()
cumulative_returns = (1 + daily_returns).cumprod()
volatility         = daily_returns.std() * np.sqrt(252)
correlation        = daily_returns.corr()

# Naive Linear‐Trend Forecast

horizon = int(input("\nEnter forecast horizon in trading days (e.g. 30): "))

for ticker in tickers:
    prices = adj_close[ticker].values
    x = np.arange(len(prices))
    # Fit a 1st‑degree polynomial (straight line)
    slope, intercept = np.polyfit(x, prices, 1)
    # Forecast price at time len(prices)+horizon
    forecast = slope * (len(prices) + horizon) + intercept
    print(f"\nNaïve {horizon}‑day forecast for {ticker}: ${forecast:,.2f}")

    #Plot last 60 days + forecast point
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

#Portfolio Allocation Recommendation

#Ask user for a total portfolio size
portfolio_value = float(input("\nEnter total portfolio value in USD (e.g. 10000): "))

#Calculate inverse‑volatility weights
inv_vol = 1 / volatility
weights_iv = inv_vol / inv_vol.sum()

#Calculate dollar allocations
allocations = weights_iv * portfolio_value

print("\nRecommended allocation (inverse-volatility weighting):")
for ticker, w in weights_iv.items():
    print(f"  {ticker}: {w:.2%} → ${allocations[ticker]:,.2f}")

#Value at Risk (VaR)

var_position = float(input("\nEnter position size for VaR calculation in USD (e.g. 1000): "))

print("\nOne-day 95% Historical VaR:")
for ticker in tickers:
    losses = -daily_returns[ticker]
    #95th percentile of losses
    var95 = np.percentile(losses, 95) * var_position
    print(f"  {ticker}: ${var95:,.2f} ({var95/var_position:.2%} of position)")


