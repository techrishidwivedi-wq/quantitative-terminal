import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

@st.cache_data(ttl=600, show_spinner=False)
def fetch_market_data(tickers_list, news_api_key=None):
    data_dict = {}
    valid_tickers = []
    
    def fetch_single(ticker_sym):
        try:
            ticker_obj = yf.Ticker(ticker_sym)
            hist = ticker_obj.history(period="max")
            info = ticker_obj.info
            
            if hist.empty:
                return None

            news_data = []
            if news_api_key:
                try:
                    params = {
                        'q': ticker_sym,
                        'apiKey': news_api_key,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 15 
                    }
                    response = requests.get(NEWS_ENDPOINT, params=params)
                    if response.status_code == 200:
                        news_data = response.json().get('articles', [])
                except Exception:
                    news_data = []
            else:
                news_data = []

            return ticker_sym, {'info': info, 'history': hist, 'news': news_data}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_single, tickers_list)
        
    for result in results:
        if result:
            ticker, data = result
            data_dict[ticker] = data
            valid_tickers.append(ticker)
            
    return data_dict, valid_tickers

def get_fundamental_df(valid_tickers, market_data):
    rows = []
    for t in valid_tickers:
        i = market_data[t]['info']

        def get_val(key, default=np.nan):
            return i.get(key, default)

        rows.append({
            'Ticker': t,
            'Mkt Cap': get_val('marketCap', 0),
            'Price': get_val('currentPrice'),
            'P/E': get_val('trailingPE'),
            'Current Ratio': get_val('currentRatio'),
            'Quick Ratio': get_val('quickRatio'),
            'Debt/Eq': get_val('debtToEquity'), 
            'Net Margin (%)': get_val('profitMargins', 0) * 100 if get_val('profitMargins') else np.nan,
            'ROE (%)': get_val('returnOnEquity', 0) * 100 if get_val('returnOnEquity') else np.nan,
            'ROA (%)': get_val('returnOnAssets', 0) * 100 if get_val('returnOnAssets') else np.nan
        })
        
    return pd.DataFrame(rows).set_index('Ticker')