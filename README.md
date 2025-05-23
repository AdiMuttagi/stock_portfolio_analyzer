# Stock Portfolio Analyzer & Backtester

## Description  
A command-line tool for fetching and analyzing historical stock data. Users can enter one or more ticker symbols to compute key metrics, visualize price behavior, forecast future prices with a simple linear trend, allocate portfolio weights by inverse volatility, and backtest forecast accuracy over a fixed out-of-sample period.


## Features  
- Fetch adjusted closing prices using yfinance  
- Calculate daily returns, cumulative returns, annualized volatility, and correlation matrix  
- Provide a naïve linear-trend forecast for a user-specified number of trading days  
- Plot recent price history and forecast point for each stock  
- Recommend portfolio allocations using inverse-volatility weighting  
- Backtest a single-stock forecast based on historical training and evaluation windows 

## Installation  
1. Clone the repository  
2. Create and activate a virtual environment  
3. Install dependencies:  
   ```bash
   pip install pandas numpy matplotlib yfinance

## Future Improvements
- Integrate more sophisticated forecasting models (ARIMA, exponential smoothing)
- Add machine‑learning‑based price prediction using scikit‑learn or TensorFlow
- Implement a graphical user interface or web dashboard (e.g. Streamlit)
- Extend backtesting to multiple assets and rolling windows for performance statistics
- Include risk metrics such as Conditional VaR and scenario analysis
- Automate data updates and scheduling for near‑real‑time monitoring
