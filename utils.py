import pandas as pd

class TechnicalAnalysis:
    @staticmethod
    def get_indicators(history: pd.DataFrame) -> pd.DataFrame:
        df = history.copy()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

def format_currency(value):
    return f"${value:,.2f}"

def format_percent(value):
    return f"{value:.2%}"