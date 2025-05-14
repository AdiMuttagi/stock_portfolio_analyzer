import pandas as pd
import numpy as np
import yfinance as yf

def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    close = df["Close"]
    if df.empty:
        print(f"No data returned for {ticker}. Possible rate-limited.")
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close

def linear_forecast(prices: pd.Series, forecast_date: pd.Timestamp) -> float:
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    y = prices.values
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)
    delta_days = (forecast_date - prices.index[-1]).days
    X = len(prices) + delta_days
    return float(m * X + b)

def main():
    ticker = input("Ticker to backtest (e.g. AAPL): ").strip().upper()

    # Training window
    train_start = "2025-01-01"
    train_end   = "2025-04-01"
    prices_train = fetch_data(ticker, train_start, train_end)

    # Forecast
    forecast_date = pd.Timestamp("2025-04-15") #Forecasting 15 days after training window
    predicted = linear_forecast(prices_train, forecast_date)

    # Actual price on or just before forecast date
    prices_full = fetch_data(ticker, train_start, "2025-04-16")
    if forecast_date in prices_full.index:
        actual = prices_full[forecast_date]
    else:
        actual = prices_full[:forecast_date].iloc[-1]


    # Error metrics
    error     = predicted - actual
    abs_error = abs(error)
    pct_error = abs_error / actual * 100

    # Report
    print("\nBacktest for", ticker)
    print("  Trained on:", train_start, "to", train_end)
    print("  Forecast date    :", forecast_date.date())
    print("  Predicted close  : $", round(predicted, 2))
    print("  Actual close     : $", round(actual, 2))
    print("  Absolute error   : $", round(abs_error, 2))
    print("  Percentage error :", round(pct_error, 2), "%")


if __name__ == "__main__":
    main()
