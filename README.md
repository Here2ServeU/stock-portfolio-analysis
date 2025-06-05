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

## Installation

```bash
pip install yfinance matplotlib pandas numpy scipy langchain-openai PyPortfolioOpt
```

## How to Use the Python Script

### Option 1: Google Colab (Recommended for Larger Datasets)
1. Upload the `stock_portfolio_analysis.py` file to your Colab environment.
2. Run each section to fetch data, analyze KPIs, and optimize portfolios.
3. Colab provides better memory and GPU support for heavier computations.

### Option 2: Jupyter Notebook
1. Launch a Jupyter Notebook.
2. Either:
   - Convert the `.py` script into a notebook, or
   - Use `%run stock_portfolio_analysis.py` in a notebook cell.
3. Useful for local data exploration and experimentation.

### Option 3: Command-Line Execution
```bash
python stock_portfolio_analysis.py
```
- This will output optimal portfolio weights using Modern Portfolio Theory.

## Notes
- Replace the placeholder API key in your LangChain setup with a valid OpenAI key if integrating LLM features.
- For enhanced analysis, use monthly aggregated data to reduce token size when using GPT.

## License

MIT License  
© 2025 Emmanuel Naweji

