"""
GenAI-Enhanced Stock Portfolio Analysis

This script uses yfinance to fetch historical stock data and calculate technical KPIs (RSI, Bollinger Bands, P/E Ratio, Beta, MACD).
It also includes functions for optimizing a portfolio using Modern Portfolio Theory (MPT).
To use LangChain/OpenAI features for prompt generation and summaries, integrate with the LangChain API.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Configuration
tickers = sorted(['AAPL', 'AMZN', 'BTC-USD', 'GOOGL', 'META', 'MSFT', 'NVDA', 'SPY', 'TSLA'])
start_date = '2022-01-01'
end_date = '2023-12-31'

# Fetch historical stock data
def fetch_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# RSI
def plot_rsi(ticker, start_date, end_date, window=14):
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, rsi, label='RSI')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title(f'RSI of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.show()

# Bollinger Bands
def plot_bollinger_bands(ticker, start_date, end_date, window=20):
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close')
    plt.plot(sma, label='SMA')
    plt.plot(upper, label='Upper Band', linestyle='--')
    plt.plot(lower, label='Lower Band', linestyle='--')
    plt.title(f'Bollinger Bands for {ticker}')
    plt.legend()
    plt.show()

# P/E Ratio
def plot_pe_ratios(tickers, start_date, end_date):
    plt.figure(figsize=(10, 5))
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        eps = stock.info.get('trailingEps', None)
        if eps and eps != 0:
            pe_ratio = data['Close'] / eps
            plt.plot(data.index, pe_ratio, label=f'{ticker}')
    plt.title('Price-to-Earnings Ratios')
    plt.xlabel('Date')
    plt.ylabel('P/E Ratio')
    plt.legend()
    plt.show()

# Beta Comparison
def plot_beta_comparison(tickers, start_date, end_date):
    betas = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        beta = stock.info.get('beta', None)
        if beta is not None:
            betas[ticker] = beta
    plt.bar(betas.keys(), betas.values())
    plt.title('Beta Comparison')
    plt.xlabel('Ticker')
    plt.ylabel('Beta Value')
    plt.show()

# MACD
def plot_macd(tickers, start_date, end_date):
    for ticker in tickers:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        plt.figure(figsize=(10, 5))
        plt.plot(macd, label='MACD')
        plt.plot(signal, label='Signal Line')
        plt.title(f'MACD for {ticker}')
        plt.legend()
        plt.show()

# MPT Optimization
def optimize_portfolio(tickers, start_date, end_date, risk_free_rate=0.04):
    data = fetch_data(tickers, start_date, end_date)
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def portfolio_perf(weights):
        ret = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, std

    def neg_sharpe(weights):
        ret, std = portfolio_perf(weights)
        return -(ret - risk_free_rate) / std if std != 0 else np.inf

    num_assets = len(tickers)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    initial = num_assets * [1. / num_assets]
    result = minimize(neg_sharpe, initial, bounds=bounds, constraints=constraints)
    weights = result.x
    return {tickers[i]: round(weights[i], 2) for i in range(num_assets)}

# Run full analysis
if __name__ == "__main__":
    print("Fetching data and performing portfolio optimization...")
    weights = optimize_portfolio(tickers, start_date, end_date)
    print("Optimal weights (MPT):", weights)
    for ticker in tickers:
        plot_rsi(ticker, start_date, end_date)
        plot_bollinger_bands(ticker, start_date, end_date)
    plot_pe_ratios(tickers, start_date, end_date)
    plot_beta_comparison(tickers, start_date, end_date)
    plot_macd(tickers, start_date, end_date)

