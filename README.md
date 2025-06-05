# Stock Portfolio Analysis with LangChain, GPT-4, YFinance, and Portfolio Optimization

This project demonstrates how to combine Generative AI (OpenAI GPT-4), LangChain, and financial data analysis libraries in Python to:
- Analyze stock performance with KPIs like RSI, MACD, Bollinger Bands, Beta, and P/E Ratio.
- Visualize stock indicators with Matplotlib.
- Optimize portfolios using Modern Portfolio Theory (MPT) and Black-Litterman Model.

## Key Concepts

### 1. LangChain + OpenAI Integration
Uses GPT-4 to:
- Generate Python code for stock data analysis.
- Summarize KPI data into an executive-level report.

### 2. YFinance for Financial Data
Fetches historical price data and fundamental metrics such as EPS and Beta.

### 3. Financial KPIs
Calculated and visualized for each stock:
- Relative Strength Index (RSI)
- Bollinger Bands
- Price-to-Earnings (P/E) Ratio
- Beta (Volatility)
- Moving Average Convergence Divergence (MACD)

### 4. Portfolio Optimization
- **Modern Portfolio Theory (MPT)** maximizes the Sharpe Ratio using historical return and covariance.
- **Black-Litterman Model** adjusts expected returns based on investor beliefs and market equilibrium.

## Use Cases
- Retail and institutional investment analysis
- AI-powered financial advisors
- Finance and data science education
- Backtesting investment strategies

## Teaching Suggestions
- Walk through KPI calculation step-by-step
- Plot each KPI for interpretation
- Encourage prompt refinement when using GPT-4
- Highlight tradeoffs and assumptions in each model

## Installation

```bash
pip install yfinance matplotlib pandas numpy scipy langchain-openai PyPortfolioOpt
```

## Run the Project

Use the Python script included in this repo:
- `stock_portfolio_analysis.py`

Ensure you replace the placeholder API key with your actual OpenAI key if using LangChain.

## License

MIT License  
© 2025 Emmanuel Naweji

