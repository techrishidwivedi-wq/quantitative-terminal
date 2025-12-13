PAGE_CONFIG = {
    "page_title": "Quantitative Terminal",
    "page_icon": ":bar_chart:",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

CUSTOM_CSS = """
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    /* Metric Cards */
    div[data-testid="stMetricValue"] { font-size: 1.4rem; color: #00ff00; }
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #444;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
"""

DEFAULT_RISK_FREE_RATE = 0.045
TRADING_DAYS = 252