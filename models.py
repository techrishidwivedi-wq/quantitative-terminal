import numpy as np
import pandas as pd
import scipy.stats as si
from datetime import datetime
from textblob import TextBlob

class PricingEngine:
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str="call"):
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == "call":
                price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
                delta = si.norm.cdf(d1)
            else:
                price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
                delta = -si.norm.cdf(-d1)
                
            gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * si.norm.pdf(d1) * np.sqrt(T) / 100 
            theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2 if option_type=="call" else -d2)) / 365
            return price, delta, gamma, theta, vega
        except: return 0,0,0,0,0

    @staticmethod
    def binomial_tree_american(S, K, T, r, sigma, N=50, option_type='call'):
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        C = np.zeros(N + 1)
        S_T = np.array([S * (u ** (N - i)) * (d ** i) for i in range(N + 1)])
        
        if option_type == 'call':
            C = np.maximum(0, S_T - K)
        else:
            C = np.maximum(0, K - S_T)
            
        for i in range(N - 1, -1, -1):
            C[:i+1] = np.exp(-r * dt) * (p * C[:i+1] + (1 - p) * C[1:i+2])
            S_t = S * (u ** (np.arange(i, -1, -1))) * (d ** (np.arange(0, i + 1)))
            if option_type == 'call':
                C[:i+1] = np.maximum(C[:i+1], S_t - K)
            else:
                C[:i+1] = np.maximum(C[:i+1], K - S_t)
        return C[0]


class PortfolioEngine:
    @staticmethod
    def simulate_efficient_frontier(prices_df, num_portfolios=2000, risk_free_rate=0.045):
        returns = prices_df.pct_change()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(prices_df.columns)
        
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            p_ret = np.sum(weights * mean_returns)
            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            results[0,i] = p_ret
            results[1,i] = p_std
            results[2,i] = (p_ret - risk_free_rate) / p_std
            
        return results, weights_record

class NewsEngine:
    @staticmethod
    def fetch_and_format_news(news_list):
        if not news_list:
            return []
        
        formatted_news = []
        
        for item in news_list:
            title = item.get('title', '')
            if not title or not title.strip():
                continue
                
            publish_time = item.get('providerPublishTime', 0)
            date_str = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
            
            formatted_news.append({
                'Title': title,
                'Publisher': item.get('publisher', 'Unknown'),
                'Link': item.get('link', '#'),
                'Date': date_str,
                'Thumbnail': item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', None)
            })
            
        return formatted_news
    
class FinancialVizEngine:
    @staticmethod
    def process_financials(ticker_obj):
        try:
            inc = ticker_obj.quarterly_income_stmt.T.iloc[:5][::-1]
            bal = ticker_obj.quarterly_balance_sheet.T.iloc[:5][::-1]
            cf = ticker_obj.quarterly_cash_flow.T.iloc[:5][::-1]
            
            perf_df = pd.DataFrame()
            if not inc.empty:
                perf_df['Revenue'] = inc.get('Total Revenue', np.nan)
                perf_df['Net Income'] = inc.get('Net Income', np.nan)
                perf_df['Net Margin'] = (perf_df['Net Income'] / perf_df['Revenue'])
                perf_df.index = [d.strftime('%b \'%y') for d in perf_df.index]

            waterfall_data = {}
            if not inc.empty:
                last_q = inc.iloc[-1]
                waterfall_data = {
                    'Revenue': last_q.get('Total Revenue', 0),
                    'COGS': -last_q.get('Cost Of Revenue', 0),
                    'Gross Profit': last_q.get('Gross Profit', 0),
                    'Op Expenses': -last_q.get('Operating Expense', 0),
                    'Op Income': last_q.get('Operating Income', 0),
                    'Tax': -last_q.get('Tax Provision', 0),
                    'Net Income': last_q.get('Net Income', 0)
                }

            debt_df = pd.DataFrame()
            if not bal.empty and not cf.empty:
                debt_df['Debt'] = bal.get('Total Debt', 0)
                debt_df['Cash'] = bal.get('Cash And Cash Equivalents', 0)
                fcf = cf.get('Free Cash Flow', pd.Series(dtype=float))
                debt_df['FCF'] = fcf.reindex(debt_df.index).fillna(0)
                debt_df.index = [d.strftime('%b \'%y') for d in debt_df.index]

            earnings_data = pd.DataFrame()
            try:
                earn = ticker_obj.earnings_dates
                if earn is not None and not earn.empty:
                    today = pd.Timestamp.now().tz_localize(earn.index.dtype.tz)
                    earn = earn.sort_index()
                    start_date = today - pd.DateOffset(months=12)
                    end_date = today + pd.DateOffset(months=6)
                    mask = (earn.index >= start_date) & (earn.index <= end_date)
                    earnings_data = earn.loc[mask].copy()
                    earnings_data.index = [d.strftime('%b \'%y') for d in earnings_data.index]
            except:
                pass

            return perf_df, waterfall_data, debt_df, earnings_data

        except Exception as e:
            return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()
        
class SentimentEngine:
    @staticmethod
    def analyze_news(news_list):
        if not news_list:
            return 0, pd.DataFrame()
        
        data = []
        total_polarity = 0
        count = 0
        
        for item in news_list:
            title = item.get('title', '')
            if not title or not title.strip():
                continue
            
            blob = TextBlob(title)
            pol = blob.sentiment.polarity
            total_polarity += pol
            count += 1
            data.append({'Title': title, 'Score': pol})
            
        if count == 0:
            return 0, pd.DataFrame(columns=['Title', 'Score'])

        avg_score = total_polarity / count
        return avg_score, pd.DataFrame(data)