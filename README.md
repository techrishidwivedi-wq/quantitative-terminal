# Financial Terminal

A comprehensive quantitative finance analysis platform built with Streamlit, providing real-time market data, portfolio management, risk analysis, backtesting, and AI-powered sentiment analysis.

## Features

- **Real-time Market Data**: Fetch live stock prices and historical data using Yahoo Finance
- **Portfolio Analysis**: Optimize portfolios with modern portfolio theory
- **Risk Management**: Calculate VaR, Sharpe ratios, and other risk metrics
- **Backtesting**: Test trading strategies against historical data
- **Technical Analysis**: Built-in technical indicators and charting
- **Sentiment Analysis**: AI-powered news sentiment analysis using NewsAPI
- **Financial Visualization**: Interactive charts and dashboards with Plotly
- **News Integration**: Latest financial news and headlines

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd quant_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- YFinance
- SciPy
- NewsAPI key (for news and sentiment features)

## Usage

1. Obtain a NewsAPI key from [newsapi.org](https://newsapi.org/)

2. Run the application:
```bash
streamlit run main.py
```

3. Open your browser to the provided URL (typically http://localhost:8501)

4. Enter ticker symbols in the sidebar and explore the various analysis tabs

## Configuration

- Modify `config.py` to customize the application appearance and default settings
- Update the NewsAPI key in `main.py` for news and sentiment features

## Project Structure

- `main.py`: Main Streamlit application
- `config.py`: Application configuration and styling
- `models.py`: Core financial engines (pricing, risk, portfolio, etc.)
- `data_engine.py`: Data fetching and processing utilities
- `utils.py`: Technical analysis and helper functions

## Disclaimer

This tool is for educational and informational purposes only. Not intended as financial advice. Always do your own research and consult with financial professionals before making investment decisions.