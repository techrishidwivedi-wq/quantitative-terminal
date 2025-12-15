import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from io import BytesIO
from scipy.stats import norm

from config import PAGE_CONFIG, CUSTOM_CSS
from models import (
    PricingEngine, 
    PortfolioEngine, 
    SentimentEngine,    
    NewsEngine,         
    FinancialVizEngine  
)
from data_engine import fetch_market_data, get_fundamental_df

st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    st.title("Quantitative Terminal")
    
    with st.sidebar:
        st.header("Configuration")
        
        try:
            news_api_key = st.secrets["news_api_key"]
        except:
            news_api_key = ""

        user_tickers = st.text_area("Tickers", "AAPL, MSFT, GOOGL, AMZN, TSLA", height=68)
        tickers_clean = [x.strip().upper() for x in user_tickers.split(',') if x.strip()]
        
        if st.button("Reload Market Data"): 
            st.cache_data.clear()
        
        st.divider()
        
        st.subheader("Market Parameters")
        try:
            tnx_data, _ = fetch_market_data(['^TNX'], None)
            if '^TNX' in tnx_data:
                market_rf = tnx_data['^TNX']['history']['Close'].iloc[-1] / 100
                st.info(f"10Y Treasury Yield: {market_rf:.2%}")
                default_rf = float(market_rf)
            else:
                default_rf = 0.045
        except:
            default_rf = 0.045
            
        rf_rate = st.number_input("Risk-Free Rate (r)", 0.01, 0.15, default_rf, 0.001)

    if not tickers_clean:
        st.info("Awaiting Input: Please enter ticker symbols.")
        return

    with st.spinner(f'Ingesting data for {len(tickers_clean)} assets...'):
        market_data, valid_tickers = fetch_market_data(tickers_clean, news_api_key)
        
    if not valid_tickers:
        st.error("Error: No valid data found. Verify ticker symbols.")
        return

    with st.sidebar:
        st.divider()
        st.header("Asset Focus")
        active_ticker = st.selectbox("Select Asset", valid_tickers)
        
        with st.expander("Latest Headlines", expanded=False):
            if not news_api_key:
                st.warning("Enter NewsAPI Key to view news.")
            else:
                news_items = NewsEngine.fetch_and_format_news(market_data[active_ticker]['news'])
                if news_items:
                    for item in news_items[:5]:
                        st.markdown(f"**[{item['Title']}]({item['Link']})**")
                        st.caption(f"{item['Date']} • {item['Publisher']}")
                else:
                    st.write("No news available.")

        st.subheader("AI Sentiment Analysis")
        if news_api_key:
            score, sent_df = SentimentEngine.analyze_news(market_data[active_ticker]['news'])
            
            if score > 0.05:
                score_color = "#4caf50"
                status_text = "↑ Bullish"
            elif score < -0.05:
                score_color = "#ef5350"
                status_text = "↓ Bearish"
            else:
                score_color = "#ff9800"
                status_text = "— Neutral"

            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 15px;">
                    <span style="font-size: 14px; opacity: 0.7;">Aggregate Score</span>
                    <div style="font-size: 36px; font-weight: bold; line-height: 1.2;">{score:.2f}</div>
                    <div style="color: {score_color}; font-weight: 600; font-size: 16px;">{status_text}</div>
                </div>
            """, unsafe_allow_html=True)
            
            if not sent_df.empty:
                dynamic_height = (len(sent_df) + 1) * 35 + 3
                
                st.dataframe(
                    sent_df[['Title', 'Score']].style
                    .format({'Score': "{:.2f}"})
                    .background_gradient(cmap='RdYlGn', subset=['Score'], vmin=-1, vmax=1),
                    height=dynamic_height, 
                    use_container_width=True, 
                    hide_index=True
                )
        else:
            st.caption("Sentiment analysis requires NewsAPI Key.")

    tabs = st.tabs([
        "Fundamentals", 
        "Financials", 
        "Portfolio", 
        "Derivatives", 
        "Risk", 
        "Technicals", 
        "Analysis", 
        "Strategy"
    ])

    with tabs[0]:
        st.subheader("Fundamental Analysis & Screening")
        
        df_fund = get_fundamental_df(valid_tickers, market_data)

        df_fund = df_fund.reset_index() 
        df_fund = df_fund.drop(columns=['ROA (%)'], errors='ignore')
        
        def color_fundamentals(val, column):
            if pd.isna(val): return ''
            
            def style(bg_color):
                text_color = 'white' if bg_color in ['#4caf50', '#ef5350'] else 'black'
                return f'background-color: {bg_color}; color: {text_color}'

            if column == 'Current Ratio':
                return style('#4caf50') if val >= 1.5 else style('#ef5350') if val < 1.0 else style('#ff9800')
                
            if column == 'Quick Ratio':
                return style('#4caf50') if val >= 1.0 else style('#ef5350') if val < 0.8 else style('#ff9800')
                
            if column == 'Debt/Eq':
                return style('#4caf50') if val < 100 else style('#ef5350') if val > 200 else style('#ff9800')

            if column == 'Net Margin (%)':
                return style('#4caf50') if val > 10 else style('#ef5350') if val < 5 else style('#ff9800')
            
            if column == 'ROE (%)':
                return style('#4caf50') if val > 15 else style('#ef5350') if val < 10 else style('#ff9800')
            
            if column == 'P/E':
                return style('#4caf50') if val < 20 else style('#ef5350') if val > 35 else style('#ff9800')

            return ''

        styled_df = df_fund.style.apply(lambda x: [color_fundamentals(v, x.name) for v in x], axis=0)
        
        styled_df = styled_df.format({
            'Mkt Cap': lambda x: f"${x/1e9:.2f}B",
            'Price': "${:.2f}",
            'P/E': "{:.1f}",
            'Current Ratio': "{:.2f}",
            'Quick Ratio': "{:.2f}",
            'Debt/Eq': "{:.1f}%",
            'Net Margin (%)': "{:.1f}%",
            'ROE (%)': "{:.1f}%",
        })

        st.dataframe(
            styled_df, 
            use_container_width=True, 
            height=(len(df_fund) + 1) * 35 + 3,
            hide_index=True
        )
        
        st.caption("🟢 Ideal/Safe | 🟠 Neutral | 🔴 Caution/Risky")
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_fund.to_excel(writer, sheet_name='Fundamentals')
        st.download_button("Export Data (.xlsx)", output.getvalue(), "fundamentals.xlsx", "application/vnd.ms-excel")

        st.divider()
        st.subheader("Market Price Movements")

        for t in valid_tickers:
            if 'history' in market_data[t] and not market_data[t]['history'].empty:
                hist = market_data[t]['history']
                
                fig = go.Figure(data=[go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name=t
                )])

                long_name = market_data[t]['info'].get('longName', '')

                fig.update_layout(
                    title=dict(
                        text=f"<b>{t} : {long_name}</b>",
                        x=0.5,
                        xanchor='center'
                    ),
                    template="plotly_dark",
                    height=450,
                    margin=dict(l=20, r=20, t=90, b=20),
                    
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=5, label="1W", step="day", stepmode="backward"),
                                dict(count=1, label="1M", step="month", stepmode="backward"),
                                dict(count=6, label="6M", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                dict(step="all", label="MAX")
                            ]),
                            font=dict(color="black", size=14), 
                            bgcolor="#e0e0e0",       
                            activecolor="#4caf50",   
                            x=0,                     
                            y=1.15
                        ),
                        type="date",
                        
                        showline=True,
                        linewidth=2,
                        linecolor='rgba(255, 255, 255, 0.3)',
                        mirror=True
                    ),
                    
                    yaxis=dict(
                        showline=True,
                        linewidth=2,
                        linecolor='rgba(255, 255, 255, 0.3)',
                        mirror=True
                    ),
                    
                    xaxis_rangeslider_visible=False 
                )

                st.plotly_chart(fig, use_container_width=True)
                
    with tabs[1]:
        st.subheader(f"Deep-Dive Financials: {active_ticker}")
        ticker_obj = yf.Ticker(active_ticker)
        perf_df, waterfall, debt_df, _ = FinancialVizEngine.process_financials(ticker_obj)
        
        if not perf_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Quarterly Performance**")
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Bar(x=perf_df.index, y=perf_df['Revenue'], name='Revenue', marker_color='#2962ff'))
                fig_perf.add_trace(go.Bar(x=perf_df.index, y=perf_df['Net Income'], name='Net Income', marker_color='#00b8d4'))
                fig_perf.add_trace(go.Scatter(x=perf_df.index, y=perf_df['Net Margin'], name='Net Margin %', yaxis='y2', line=dict(color='#ff6d00', width=3)))
                fig_perf.update_layout(yaxis2=dict(overlaying='y', side='right', tickformat='.1%'), legend=dict(orientation="h", y=1.1), height=350, template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_perf, use_container_width=True)
            with c2:
                st.markdown("**Revenue to Profit Conversion (MRQ)**")
                if waterfall:
                    fig_water = go.Figure(go.Waterfall(
                        measure = ["relative", "relative", "total", "relative", "total", "relative", "total"],
                        x = ["Rev", "COGS", "Gross", "OpExp", "OpInc", "Tax", "Net"],
                        textposition = "outside",
                        y = [waterfall['Revenue'], waterfall['COGS'], waterfall['Gross Profit'], waterfall['Op Expenses'], waterfall['Op Income'], waterfall['Tax'], waterfall['Net Income']],
                        connector = {"line":{"color":"rgb(63, 63, 63)"}},
                        decreasing = {"marker":{"color":"#ef5350"}},
                        increasing = {"marker":{"color":"#26a69a"}},
                        totals = {"marker":{"color":"#42a5f5"}}
                    ))
                    fig_water.update_layout(height=350, template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_water, use_container_width=True)
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**Debt Level & Cash Flow**")
                if not debt_df.empty:
                    fig_debt = go.Figure()
                    fig_debt.add_trace(go.Bar(x=debt_df.index, y=debt_df['Debt'], name='Total Debt', marker_color='#ef5350'))
                    fig_debt.add_trace(go.Bar(x=debt_df.index, y=debt_df['FCF'], name='Free Cash Flow', marker_color='#29b6f6'))
                    fig_debt.add_trace(go.Bar(x=debt_df.index, y=debt_df['Cash'], name='Cash & Equiv', marker_color='#66bb6a'))
                    fig_debt.update_layout(barmode='group', height=350, template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig_debt, use_container_width=True)
            with c4:
                st.markdown("**Quality of Earnings (Cash vs Profit)**")
                if not perf_df.empty and not debt_df.empty:
                    fig_qoe = go.Figure()
                    
                    fig_qoe.add_trace(go.Bar(
                        x=perf_df.index, 
                        y=perf_df['Net Income'], 
                        name='Net Income', 
                        marker_color='#2962ff'
                    ))
                    
                    fig_qoe.add_trace(go.Bar(
                        x=debt_df.index, 
                        y=debt_df['FCF'], 
                        name='Free Cash Flow', 
                        marker_color='#00c853'
                    ))

                    fig_qoe.update_layout(
                        barmode='group',
                        height=350, 
                        template='plotly_dark', 
                        margin=dict(l=20, r=20, t=40, b=20),
                        legend=dict(orientation="h", y=1.1),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_qoe, use_container_width=True)
                else:
                    st.write("Insufficient data for Quality of Earnings.")
        else:
            st.warning("Detailed financial data unavailable for this asset via API.")

    with tabs[2]:
        st.subheader("Modern Portfolio Theory")
        if len(valid_tickers) < 2:
            st.warning("Select at least 2 tickers for optimization.")
        else:
            prices = pd.DataFrame({t: market_data[t]['history']['Close'] for t in valid_tickers})
            results, weights = PortfolioEngine.simulate_efficient_frontier(prices, 2000, rf_rate)
            max_sharpe_idx = np.argmax(results[2])
            c1, c2 = st.columns([3, 1])
            with c1:
                hover_text = []
                for w in weights:
                    sorted_indices = np.argsort(w)[::-1][:3]
                    top_holdings = [f"{prices.columns[i]}: {w[i]:.1%}" for i in sorted_indices]
                    hover_text.append("<br>" + "<br>".join(top_holdings))

                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=results[1,:], 
                    y=results[0,:], 
                    mode='markers', 
                    text=hover_text,
                    hovertemplate="<b>Return:</b> %{y:.1%}<br><b>Vol:</b> %{x:.1%}<br><b>Sharpe:</b> %{marker.color:.2f}%{text}<extra></extra>",
                    marker=dict(
                        color=results[2,:], 
                        colorscale='RdYlGn',
                        showscale=True, 
                        size=7,
                        opacity=0.7,
                        line=dict(width=0.5, color='white'),
                        colorbar=dict(title="Sharpe Ratio")
                    ), 
                    name='Portfolios'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[results[1,max_sharpe_idx]], 
                    y=[results[0,max_sharpe_idx]], 
                    mode='markers', 
                    marker=dict(
                        color='gold',
                        size=22, 
                        symbol='star',
                        line=dict(width=2, color='black')
                    ), 
                    name='Max Sharpe',
                    hoverinfo='skip'
                ))

                fig.add_annotation(
                    x=results[1,max_sharpe_idx],
                    y=results[0,max_sharpe_idx],
                    text="Best Risk-Adj. Return",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40,
                    font=dict(color="white", size=12),
                    bgcolor="#333333",
                    opacity=0.8
                )

                fig.update_layout(
                    template="plotly_dark", 
                    xaxis_title="Volatility (Risk)", 
                    yaxis_title="Expected Return",
                    height=500,
                    margin=dict(l=40, r=40, t=40, b=40),
                    legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'),
                    
                    xaxis=dict(
                        showline=True,
                        linewidth=2,
                        linecolor='rgba(255, 255, 255, 0.3)',
                        mirror=True
                    ),
                    yaxis=dict(
                        showline=True,
                        linewidth=2,
                        linecolor='rgba(255, 255, 255, 0.3)',
                        mirror=True
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(pd.DataFrame(weights[max_sharpe_idx], index=prices.columns, columns=['Weight']).style.format("{:.1%}"), use_container_width=True)

    with tabs[3]:
        st.subheader("Derivatives & Option Pricing")
        
        col_input, col_viz = st.columns([1, 2])
        
        with col_input:
            st.markdown("### Parameters")
            current_price = market_data[active_ticker]['history']['Close'].iloc[-1]
            
            S = st.number_input("Spot Price (S)", value=float(current_price), step=1.0)
            K = st.number_input("Strike Price (K)", value=float(current_price), step=1.0)
            T = st.slider("Time to Maturity (Years)", 0.01, 2.0, 0.5, 0.01)
            sigma = st.slider("Volatility (σ)", 0.05, 1.5, 0.30, 0.01)
            r = st.number_input("Risk-Free Rate", value=rf_rate, format="%.4f", disabled=True)
            
            steps = st.slider("Binomial Steps (N)", 10, 200, 50, 10, help="Higher steps = More accurate.")
            
            option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)

        with col_viz:
            bs_price, d, g, th, v = PricingEngine.black_scholes(S, K, T, r, sigma, option_type.lower())
            bn_price = PricingEngine.binomial_tree_american(S, K, T, r, sigma, N=steps, option_type=option_type.lower())
            
            diff = bn_price - bs_price
            intrinsic = max(0, S-K) if option_type == "Call" else max(0, K-S)

            st.subheader("Derivatives Analytics")

            analytics_data = [
                {"Metric": "Black-Scholes (European)", "Value": f"${bs_price:.2f}", "Notes": "Standard Model"},
                {"Metric": "Binomial (American)",     "Value": f"${bn_price:.2f}", "Notes": f"Spread: {diff:+.2f}"},
                {"Metric": "Intrinsic Value",     "Value": f"${intrinsic:.2f}","Notes": "Floor Value"},
                {"Metric": "Delta (Δ)",           "Value": f"{d:.4f}",         "Notes": " exposure to Spot"},
                {"Metric": "Gamma (Γ)",           "Value": f"{g:.4f}",         "Notes": " convexity"},
                {"Metric": "Theta (Θ)",           "Value": f"{th:.4f}",        "Notes": " daily time decay"},
                {"Metric": "Vega (ν)",            "Value": f"{v:.4f}",         "Notes": " exposure to Vol"},
            ]

            df_analytics = pd.DataFrame(analytics_data)

            st.dataframe(
                df_analytics.style.applymap(
                    lambda x: "background-color: #263238; color: #69f0ae; font-weight: bold; border: 1px solid #546e7a", 
                    subset=['Value']
                ),
                hide_index=True,
                use_container_width=True,
                height=(len(df_analytics) + 1) * 35 + 3
            )

        st.divider()

        v1, v2 = st.columns(2)
        
        with v1:
            st.subheader("Strategy PnL Profile")
            
            strat_choice = st.selectbox("Simulate Strategy", ["Single Option", "Straddle (Long Vol)", "Strangle (Wide Vol)"], label_visibility="collapsed")
            
            s_range = np.linspace(S * 0.5, S * 1.5, 100)
            
            if strat_choice == "Single Option":
                if option_type == "Call":
                    payoff = np.maximum(s_range - K, 0) - bs_price
                else:
                    payoff = np.maximum(K - s_range, 0) - bs_price
                chart_title = f"{option_type} PnL"
                
            elif strat_choice == "Straddle (Long Vol)":
                call_p, _, _, _, _ = PricingEngine.black_scholes(S, K, T, r, sigma, "call")
                put_p, _, _, _, _ = PricingEngine.black_scholes(S, K, T, r, sigma, "put")
                cost = call_p + put_p
                payoff = np.maximum(s_range - K, 0) + np.maximum(K - s_range, 0) - cost
                chart_title = "Long Straddle PnL (Bet on High Volatility)"
                
            elif strat_choice == "Strangle (Wide Vol)":
                k_put = K * 0.95
                k_call = K * 1.05
                c_p, _, _, _, _ = PricingEngine.black_scholes(S, k_call, T, r, sigma, "call")
                p_p, _, _, _, _ = PricingEngine.black_scholes(S, k_put, T, r, sigma, "put")
                cost = c_p + p_p
                payoff = np.maximum(s_range - k_call, 0) + np.maximum(k_put - s_range, 0) - cost
                chart_title = f"Long Strangle PnL ({k_put:.0f}/{k_call:.0f})"

            fig_payoff = go.Figure()
            fig_payoff.add_trace(go.Scatter(
                x=s_range, y=payoff, 
                name='PnL', 
                fill='tozeroy',
                line=dict(color='#00e676' if payoff[len(payoff)//2] > 0 else '#ff1744') # Green if profitable now, else Red
            ))
            fig_payoff.add_vline(x=S, line_dash="dash", line_color="white", annotation_text="Current")
            fig_payoff.add_hline(y=0, line_color="gray")
            
            fig_payoff.update_layout(
                template="plotly_dark", title=chart_title,
                height=350, margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showline=True, linewidth=2, linecolor='rgba(255,255,255,0.3)', mirror=True),
                yaxis=dict(showline=True, linewidth=2, linecolor='rgba(255,255,255,0.3)', mirror=True)
            )
            st.plotly_chart(fig_payoff, use_container_width=True)

        with v2:
            st.subheader("Price Sensitivity (Heatmap)")
            vol_range = np.linspace(max(0.1, sigma - 0.15), sigma + 0.15, 10)
            spot_range = np.linspace(S * 0.85, S * 1.15, 10)
            z_values = [[PricingEngine.black_scholes(s_, K, T, r, v_, option_type.lower())[0] for s_ in spot_range] for v_ in vol_range]
            
            fig_heat = go.Figure(data=go.Heatmap(z=z_values, x=spot_range, y=vol_range, colorscale='Viridis'))
            fig_heat.update_layout(
                template="plotly_dark", title="Price vs Spot & Vol",
                xaxis_title="Spot Price ($)", yaxis_title="Volatility (σ)",
                height=350, margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showline=True, linewidth=2, linecolor='rgba(255,255,255,0.3)', mirror=True),
                yaxis=dict(showline=True, linewidth=2, linecolor='rgba(255,255,255,0.3)', mirror=True)
            )
            st.plotly_chart(fig_heat, use_container_width=True)

             

        st.divider()
        g_col1, g_col2 = st.columns([1, 3])
        with g_col1:
            st.markdown("### Greeks Analysis")
            greek_selection = st.selectbox("Select Greek", ["Delta (Δ)", "Gamma (Γ)", "Theta (Θ)", "Vega (ν)"])
        
        with g_col2:
            spot_range = np.linspace(S * 0.5, S * 1.5, 100)
            greek_values = []
            for s_sim in spot_range:
                _, d, g, th, v = PricingEngine.black_scholes(s_sim, K, T, r, sigma, option_type.lower())
                if "Delta" in greek_selection: val = d
                elif "Gamma" in greek_selection: val = g
                elif "Theta" in greek_selection: val = th
                else: val = v
                greek_values.append(val)

            fig_greek = go.Figure()
            fig_greek.add_trace(go.Scatter(x=spot_range, y=greek_values, mode='lines', line=dict(color='#29b6f6', width=3)))
            fig_greek.add_vline(x=S, line_dash="dash", line_color="white", annotation_text="Current")
            fig_greek.update_layout(
                title=f"<b>{greek_selection} Sensitivity</b> vs Spot Price",
                xaxis_title="Spot Price ($)", yaxis_title=greek_selection,
                template="plotly_dark", height=400, margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(showline=True, linewidth=2, linecolor='rgba(255,255,255,0.3)', mirror=True),
                yaxis=dict(showline=True, linewidth=2, linecolor='rgba(255,255,255,0.3)', mirror=True)
            )
            st.plotly_chart(fig_greek, use_container_width=True, key="greek_chart_viz")
        
        st.divider()

        st.subheader("Advanced Risk Analysis")
        
        adv_tab1, adv_tab2 = st.tabs(["Probability Cone", "3D Volatility Surface"])
        
        with adv_tab1:
            c_prob1, c_prob2 = st.columns([1, 3])
            
            with c_prob1:
                st.markdown("#### Market Implied Odds")
                d1_val = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2_val = d1_val - sigma * np.sqrt(T)
                
                if option_type == "Call":
                    prob_itm = norm.cdf(d2_val)
                else:
                    prob_itm = norm.cdf(-d2_val)
                    
                st.metric("Prob. ITM", f"{prob_itm:.1%}", help="Probability expiring In-The-Money")
                st.metric("Prob. OTM", f"{1-prob_itm:.1%}", help="Probability expiring Worthless")
                
            with c_prob2:
                expected_price = S * np.exp(r * T)
                std_dev_price = S * sigma * np.sqrt(T)
                x_axis = np.linspace(S - 3*std_dev_price, S + 3*std_dev_price, 200)
                y_axis = norm.pdf(x_axis, expected_price, std_dev_price)
                
                fig_prob = go.Figure()
                fig_prob.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', name='Probability', line=dict(color='cyan')))
                
                if option_type == "Call":
                    mask = x_axis >= K
                else:
                    mask = x_axis <= K
                fig_prob.add_trace(go.Scatter(x=x_axis[mask], y=y_axis[mask], fill='tozeroy', mode='none', fillcolor='rgba(0, 230, 118, 0.3)', name='ITM'))
                
                fig_prob.add_vline(x=K, line_dash="dash", annotation_text=f"Strike ${K}")
                fig_prob.update_layout(
                    template="plotly_dark", title="Price Distribution at Expiration",
                    height=300, showlegend=False, margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        with adv_tab2:
            st.info("Interactive Surface: Drag to rotate. See how price changes with Spot and Volatility.")
            
            spot_3d = np.linspace(S * 0.7, S * 1.3, 20)
            vol_3d = np.linspace(0.1, 0.8, 20)
            X, Y = np.meshgrid(spot_3d, vol_3d)
            
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    p_val, _, _, _, _ = PricingEngine.black_scholes(X[i,j], K, T, r, Y[i,j], option_type.lower())
                    Z[i,j] = p_val

            fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
            fig_3d.update_layout(
                template="plotly_dark",
                scene=dict(
                    xaxis_title='Spot Price ($)',
                    yaxis_title='Volatility (σ)',
                    zaxis_title='Option Price ($)'
                ),
                height=500, margin=dict(l=0, r=0, b=0, t=0)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    with tabs[4]:
        st.subheader(f"Risk Profile: {active_ticker}")
        
        hist_prices = market_data[active_ticker]['history']['Close']
        returns = hist_prices.pct_change().dropna()
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        max_drawdown = (hist_prices / hist_prices.cummax() - 1).min()
        ann_vol = returns.std() * np.sqrt(252)
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Value at Risk (95%)", f"{var_95:.2%}", help="Worst expected loss 95% of the time.")
        r2.metric("CVaR (Expected Shortfall)", f"{cvar_95:.2%}", help="Average loss on the worst 5% of days.")
        r3.metric("Max Drawdown", f"{max_drawdown:.2%}", help="Maximum observed loss from a peak to a trough.")
        r4.metric("Annualized Volatility", f"{ann_vol:.2%}", help="Standard deviation of returns scaled to 1 year.")
        
        st.divider()

        risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Underwater Plot", "Return Distribution", "Rolling Volatility"])
        
        border_settings = dict(
            showline=True, 
            linewidth=2, 
            linecolor='rgba(255, 255, 255, 0.3)', 
            mirror=True
        )

        with risk_tab1:
            drawdown_series = (hist_prices / hist_prices.cummax() - 1)
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown_series.index, 
                y=drawdown_series, 
                fill='tozeroy', 
                fillcolor='rgba(239, 83, 80, 0.3)',
                line=dict(color='#ef5350'),
                name='Drawdown'
            ))
            fig_dd.update_layout(
                title="Underwater Plot (Drawdown from Peak)",
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                height=400,
                margin=dict(l=20, r=40, t=40, b=20),
                xaxis=border_settings,
                yaxis=border_settings
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with risk_tab2:
            fig_hist = px.histogram(
                returns, 
                nbins=100, 
                title="Distribution of Daily Returns",
                color_discrete_sequence=['#29b6f6']
            )
            fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="VaR 95%")
            fig_hist.update_layout(template="plotly_dark", showlegend=False, height=400, margin=dict(l=20, r=40, t=40, b=20), xaxis=border_settings, yaxis=border_settings)
            st.plotly_chart(fig_hist, use_container_width=True) 
            
        with risk_tab3:
            window = st.slider("Rolling Window (Days)", 10, 90, 30)
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index, 
                y=rolling_vol, 
                mode='lines',
                line=dict(color='#ffa726'),
                name='Volatility'
            ))
            fig_vol.update_layout(
                title=f"{window}-Day Rolling Volatility (Annualized)",
                yaxis_title="Volatility",
                template="plotly_dark",
                height=400,
                margin=dict(l=20, r=40, t=40, b=20),
                xaxis=border_settings,
                yaxis=border_settings
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        st.divider()

        st.subheader("Cross-Asset Risk Comparison")
        
        risk_summary = []
        for t in valid_tickers:
            h = market_data[t]['history']['Close']
            r = h.pct_change().dropna()
            
            risk_summary.append({
                "Ticker": t,
                "VaR 95%": np.percentile(r, 5),
                "CVaR 95%": r[r <= np.percentile(r, 5)].mean(),
                "Max DD": (h / h.cummax() - 1).min(),
                "Vol (Ann)": r.std() * np.sqrt(252)
            })
            
        df_risk = pd.DataFrame(risk_summary)
        
        st.dataframe(
            df_risk.style.format({
                "VaR 95%": "{:.2%}", 
                "CVaR 95%": "{:.2%}", 
                "Max DD": "{:.2%}", 
                "Vol (Ann)": "{:.2%}"
            }).background_gradient(cmap='RdYlGn_r', subset=['VaR 95%', 'CVaR 95%', 'Max DD', 'Vol (Ann)']), 
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Run Monte Carlo Simulation", expanded=False):
            st.markdown("### Future Price Projection (GBM Model)")
            
            mc_col1, mc_col2, mc_col3 = st.columns(3)
            with mc_col1:
                sim_days = st.number_input("Forecast Horizon (Days)", 30, 365, 252)
            with mc_col2:
                n_sims = st.number_input("Number of Simulations", 100, 2000, 500)
            with mc_col3:
                vol_adj = st.slider("Volatility Adjustment", 0.5, 2.0, 1.0, 0.1, help="Multiply current volatility to simulate calmer/crazier markets.")

            if st.button("Run Simulation", type="primary"):
                last_price = hist_prices.iloc[-1]
                daily_vol = returns.std() * vol_adj
                daily_drift = returns.mean() 
                
                dt = 1
                shock = daily_drift - 0.5 * daily_vol**2
                random_component = daily_vol * np.random.normal(0, 1, (sim_days, n_sims))
                
                log_returns = np.cumsum(shock + random_component, axis=0)
                price_paths = last_price * np.exp(log_returns)
                
                price_paths = np.vstack([np.full(n_sims, last_price), price_paths])
                
                final_prices = price_paths[-1]
                expected_price = np.mean(final_prices)
                sigma_price = np.std(final_prices)
                worst_case = np.percentile(final_prices, 5)
                best_case = np.percentile(final_prices, 95)
                prob_profit = np.mean(final_prices > last_price)

                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Expected Price", f"${expected_price:.2f}", delta=f"{(expected_price/last_price - 1):.2%}")
                m2.metric("Worst Case (5%)", f"${worst_case:.2f}", delta=f"{(worst_case/last_price - 1):.2%}", delta_color="inverse")
                m3.metric("Best Case (95%)", f"${best_case:.2f}", delta=f"{(best_case/last_price - 1):.2%}")
                m4.metric("Prob. of Profit", f"{prob_profit:.1%}", help="Likelihood of price being higher than today.")

                st.divider()
                st.subheader("Visual Analysis")

                p5 = np.percentile(price_paths, 5, axis=1)
                p50 = np.percentile(price_paths, 50, axis=1)
                p95 = np.percentile(price_paths, 95, axis=1)
                x_axis = np.arange(sim_days + 1)

                fig_paths = go.Figure()

                fig_paths.add_trace(go.Scatter(
                    x=x_axis, y=p95, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig_paths.add_trace(go.Scatter(
                    x=x_axis, y=p5, mode='lines', line=dict(width=0), 
                    fill='tonexty', fillcolor='rgba(41, 182, 246, 0.15)',
                    name='90% Confidence Interval'
                ))

                for i in range(min(100, n_sims)): 
                    fig_paths.add_trace(go.Scatter(
                        x=x_axis, y=price_paths[:, i], 
                        mode='lines', 
                        line=dict(color='rgba(255, 255, 255, 0.15)', width=1),
                        showlegend=False, hoverinfo='skip'
                    ))

                fig_paths.add_trace(go.Scatter(
                    x=x_axis, y=p50, mode='lines', 
                    name='Median Projection',
                    line=dict(color='#ffeb3b', width=2)
                ))

                fig_paths.update_layout(
                    title=f"Projected Price Cone ({sim_days} Days)",
                    xaxis_title="Trading Days", 
                    yaxis_title="Price ($)",
                    template="plotly_dark", 
                    height=450,
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode="x unified",
                    xaxis=border_settings,
                    yaxis=border_settings
                )
                st.plotly_chart(fig_paths, use_container_width=True)


                fig_dist = go.Figure()

                profit_prices = final_prices[final_prices >= last_price]
                loss_prices = final_prices[final_prices < last_price]

                fig_dist.add_trace(go.Histogram(
                    x=loss_prices, 
                    marker_color='#ef5350',
                    name='Loss',
                    opacity=0.75,
                    xbins=dict(size=(max(final_prices)-min(final_prices))/50)
                ))
                fig_dist.add_trace(go.Histogram(
                    x=profit_prices, 
                    marker_color='#66bb6a',
                    name='Profit',
                    opacity=0.75,
                    xbins=dict(size=(max(final_prices)-min(final_prices))/50)
                ))

                fig_dist.add_vline(x=last_price, line_dash="solid", line_width=2, line_color="white", annotation_text="Start")
                fig_dist.add_vline(x=worst_case, line_dash="dash", line_color="#ef5350", annotation_text="Worst 5%")
                fig_dist.add_vline(x=best_case, line_dash="dash", line_color="#66bb6a", annotation_text="Best 5%")

                fig_dist.update_layout(
                    title="Terminal Price Distribution (Profit vs Loss)",
                    xaxis_title="Final Price ($)", 
                    yaxis_title="Frequency",
                    template="plotly_dark", 
                    height=400,
                    barmode='overlay',
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=border_settings,
                    yaxis=border_settings
                )
                st.plotly_chart(fig_dist, use_container_width=True)

    with tabs[5]:
        st.subheader(f"Technical Dashboard: {active_ticker}")
        
        hist = market_data[active_ticker]['history'].copy()
        
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
        
        hist['BB_Mid'] = hist['Close'].rolling(window=20).mean()
        hist['BB_Std'] = hist['Close'].rolling(window=20).std()
        hist['BB_Upper'] = hist['BB_Mid'] + (2 * hist['BB_Std'])
        hist['BB_Lower'] = hist['BB_Mid'] - (2 * hist['BB_Std'])
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = exp1 - exp2
        hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['MACD_Hist'] = hist['MACD'] - hist['Signal_Line']

        with st.expander("Chart Overlay Settings", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            show_sma = c1.checkbox("Show SMA 50/200", value=True)
            show_bb = c2.checkbox("Show Bollinger Bands", value=True)
            show_volume = c3.checkbox("Show Volume Profile", value=False)
            lookback = c4.select_slider("Zoom (Months)", options=[1, 3, 6, 12, 24, 60], value=12)

        subset = hist.iloc[-lookback*21:]

        from plotly.subplots import make_subplots
        
        fig_ta = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{active_ticker} Price Action", "RSI Momentum", "MACD Trend")
        )

        fig_ta.add_trace(go.Candlestick(
            x=subset.index, open=subset['Open'], high=subset['High'],
            low=subset['Low'], close=subset['Close'], name='OHLC'
        ), row=1, col=1)

        if show_sma:
            fig_ta.add_trace(go.Scatter(x=subset.index, y=subset['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
            fig_ta.add_trace(go.Scatter(x=subset.index, y=subset['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'), row=1, col=1)

        if show_bb:
            fig_ta.add_trace(go.Scatter(x=subset.index, y=subset['BB_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
            fig_ta.add_trace(go.Scatter(x=subset.index, y=subset['BB_Lower'], line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', name='Bollinger Band'), row=1, col=1)

        fig_ta.add_trace(go.Scatter(x=subset.index, y=subset['RSI'], line=dict(color='#ce93d8', width=2), name='RSI'), row=2, col=1)
        fig_ta.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig_ta.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        colors = ['#66bb6a' if v >= 0 else '#ef5350' for v in subset['MACD_Hist']]
        fig_ta.add_trace(go.Bar(x=subset.index, y=subset['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=3, col=1)
        fig_ta.add_trace(go.Scatter(x=subset.index, y=subset['MACD'], line=dict(color='#29b6f6', width=1), name='MACD'), row=3, col=1)
        fig_ta.add_trace(go.Scatter(x=subset.index, y=subset['Signal_Line'], line=dict(color='#ffa726', width=1), name='Signal'), row=3, col=1)

        border_settings = dict(showline=True, linewidth=2, linecolor='rgba(255, 255, 255, 0.3)', mirror=True)
        fig_ta.update_layout(
            template="plotly_dark", height=800, 
            margin=dict(l=20, r=40, t=40, b=20),
            xaxis=border_settings, yaxis=border_settings,
            xaxis3=border_settings, yaxis3=border_settings,
            showlegend=False
        )
        fig_ta.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig_ta, use_container_width=True)

        st.divider()

        st.subheader("Automated Signals")
        
        curr_close = subset['Close'].iloc[-1]
        curr_rsi = subset['RSI'].iloc[-1]
        curr_macd = subset['MACD'].iloc[-1]
        curr_sig = subset['Signal_Line'].iloc[-1]
        curr_upper = subset['BB_Upper'].iloc[-1]
        curr_lower = subset['BB_Lower'].iloc[-1]
        
        signals = []
        
        if curr_rsi > 70: signals.append({"Indicator": "RSI (14)", "Value": f"{curr_rsi:.1f}", "Signal": "Overbought (Sell)", "Bias": "Bearish"})
        elif curr_rsi < 30: signals.append({"Indicator": "RSI (14)", "Value": f"{curr_rsi:.1f}", "Signal": "Oversold (Buy)", "Bias": "Bullish"})
        else: signals.append({"Indicator": "RSI (14)", "Value": f"{curr_rsi:.1f}", "Signal": "Neutral", "Bias": "Neutral"})
        
        if curr_macd > curr_sig: signals.append({"Indicator": "MACD", "Value": f"{curr_macd:.2f}", "Signal": "Bullish Crossover", "Bias": "Bullish"})
        else: signals.append({"Indicator": "MACD", "Value": f"{curr_macd:.2f}", "Signal": "Bearish Divergence", "Bias": "Bearish"})
        
        if curr_close > curr_upper: signals.append({"Indicator": "Bollinger Bands", "Value": f"{curr_close:.2f}", "Signal": "Price > Upper Band", "Bias": "Bearish (Mean Rev)"})
        elif curr_close < curr_lower: signals.append({"Indicator": "Bollinger Bands", "Value": f"{curr_close:.2f}", "Signal": "Price < Lower Band", "Bias": "Bullish (Mean Rev)"})
        else: signals.append({"Indicator": "Bollinger Bands", "Value": "Within Bands", "Signal": "Range Bound", "Bias": "Neutral"})

        sma_50 = subset['SMA_50'].iloc[-1]
        if curr_close > sma_50: signals.append({"Indicator": "Trend (SMA 50)", "Value": f"{sma_50:.2f}", "Signal": "Price > SMA", "Bias": "Bullish"})
        else: signals.append({"Indicator": "Trend (SMA 50)", "Value": f"{sma_50:.2f}", "Signal": "Price < SMA", "Bias": "Bearish"})

        df_signals = pd.DataFrame(signals)
        
        def signal_color(val):
            if "Bullish" in val: return 'color: #66bb6a; font-weight: bold'
            if "Bearish" in val: return 'color: #ef5350; font-weight: bold'
            return 'color: white'

        st.dataframe(
            df_signals.style.applymap(signal_color, subset=['Bias', 'Signal']),
            use_container_width=True,
            hide_index=True
        )

    with tabs[6]:
        st.subheader(f"Quantitative Analysis: {active_ticker}")
        
        hist = market_data[active_ticker]['history']
        rets = hist['Close'].pct_change().dropna()
        
        mean_ret = rets.mean() * 252
        std_dev = rets.std() * np.sqrt(252)
        skew = rets.skew()
        kurt = rets.kurtosis()
        
        st.markdown("#### Statistical Moments")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Annualized Return", f"{mean_ret:.2%}")
        m2.metric("Annualized Vol", f"{std_dev:.2%}")
        m3.metric("Skewness", f"{skew:.2f}", help="Negative = Frequent small gains, few extreme losses.")
        m4.metric("Kurtosis", f"{kurt:.2f}", help="High > 3.0 means 'Fat Tails' (extreme events are likely).")
        
        st.divider()

        q_tab1, q_tab2, q_tab3 = st.tabs(["Seasonality", "Trend Regression", "Correlations"])

        with q_tab1:
            st.markdown("### Monthly Return Matrix")
            
            season_df = pd.DataFrame({'Return': rets})
            season_df['Year'] = season_df.index.year
            season_df['Month_Num'] = season_df.index.month
            
            pivot_ret = season_df.pivot_table(index='Year', columns='Month_Num', values='Return')
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_ret = pivot_ret.reindex(columns=range(1, 13)) 
            pivot_ret.columns = month_names 
            
            limit = max(abs(pivot_ret.min().min()), abs(pivot_ret.max().max())) * 0.7 
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_ret.values,
                x=month_names,
                y=pivot_ret.index,
                colorscale='RdYlGn', 
                zmid=0,
                zmin=-limit,
                zmax=limit,
                text=pivot_ret.values,
                texttemplate="%{z:.1%}",
                textfont={"size": 11},
                xgap=4,
                ygap=4
            ))
            
            border_settings = dict(showline=True, linewidth=2, linecolor='rgba(255, 255, 255, 0.2)', mirror=True)
            
            fig_heatmap.update_layout(
                title=dict(text=f"<b>{active_ticker} Historical Returns</b>", x=0, font=dict(size=18)),
                template="plotly_dark",
                height=550,
                
                yaxis=dict(
                    autorange="reversed", 
                    tickfont=dict(size=13, family="Arial Black"), 
                    showgrid=False,
                    title=None,
                    **border_settings
                ),
                
                xaxis=dict(
                    side="top",
                    tickfont=dict(size=13, family="Arial Black"),
                    showgrid=False,
                    title=None,
                    **border_settings
                ),
                margin=dict(l=20, r=20, t=80, b=20)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                pivot_ret.to_excel(writer, sheet_name='Seasonality')
                writer.sheets['Seasonality'].set_column(1, 12, None, writer.book.add_format({'num_format': '0.00%'}))
            buffer.seek(0)
            
            col_dl, _ = st.columns([1, 4])
            with col_dl:
                st.download_button(
                    label="Download Excel Report",
                    data=buffer,
                    file_name=f"{active_ticker}_seasonality_matrix.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )

        with q_tab2:
            st.markdown("**Linear Regression Channel**")
            
            y = hist['Close'].values
            x = np.arange(len(y))
            
            m, c = np.polyfit(x, y, 1)
            reg_line = m*x + c
            std_error = np.std(y - reg_line)
            
            fig_reg = go.Figure()
            
            fig_reg.add_trace(go.Scatter(x=hist.index, y=y, name='Price', line=dict(color='white', width=1)))
            
            fig_reg.add_trace(go.Scatter(x=hist.index, y=reg_line, name='Trend', line=dict(color='orange', dash='dash')))
            
            upper = reg_line + (2 * std_error)
            lower = reg_line - (2 * std_error)
            
            fig_reg.add_trace(go.Scatter(x=hist.index, y=upper, mode='lines', line=dict(width=0), showlegend=False))
            fig_reg.add_trace(go.Scatter(
                x=hist.index, y=lower, 
                mode='lines', line=dict(width=0), 
                fill='tonexty', fillcolor='rgba(255, 165, 0, 0.15)', 
                name='2 Std Dev Channel'
            ))
            
            fig_reg.update_layout(
                title=f"Price vs Trend (Slope: {m:.4f})",
                template="plotly_dark",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showline=True, linewidth=2, linecolor='rgba(255,255,255,0.3)', mirror=True),
                yaxis=dict(showline=True, linewidth=2, linecolor='rgba(255,255,255,0.3)', mirror=True)
            )
            st.plotly_chart(fig_reg, use_container_width=True)

        with q_tab3:
            if len(valid_tickers) < 2:
                st.warning("Select at least 2 tickers to analyze correlations.")
            else:
                st.markdown("### Dynamic Correlation Analysis")
                
                other_tickers = [t for t in valid_tickers if t != active_ticker]
                if other_tickers:
                    compare_asset = st.selectbox("Correlate against:", other_tickers)
                    
                    df_comb = pd.DataFrame({
                        'Asset_A': market_data[active_ticker]['history']['Close'],
                        'Asset_B': market_data[compare_asset]['history']['Close']
                    }).pct_change().dropna()
                    
                    window = 60
                    rolling_corr = df_comb['Asset_A'].rolling(window).corr(df_comb['Asset_B'])
                    
                    fig_corr = go.Figure()
                    
                    fig_corr.add_trace(go.Scatter(
                        x=rolling_corr.index, 
                        y=rolling_corr, 
                        fill='tozeroy',
                        line=dict(color='#ab47bc', width=2),
                        name=f'60-Day Corr'
                    ))
                    
                    fig_corr.add_hline(y=1, line_dash="dot", line_color="#66bb6a", annotation_text="Perfect Pos")
                    fig_corr.add_hline(y=0, line_dash="solid", line_color="gray", annotation_text="Uncorrelated")
                    fig_corr.add_hline(y=-1, line_dash="dot", line_color="#ef5350", annotation_text="Perfect Neg")
                    
                    border_settings = dict(showline=True, linewidth=2, linecolor='rgba(255, 255, 255, 0.3)', mirror=True)
                    
                    fig_corr.update_layout(
                        title=f"<b>Rolling correlation ({window}-Day)</b>: {active_ticker} vs {compare_asset}",
                        yaxis_title="Correlation (-1 to +1)",
                        template="plotly_dark",
                        height=450,
                        margin=dict(l=20, r=20, t=50, b=20),
                        yaxis=dict(range=[-1.1, 1.1], **border_settings),
                        xaxis=border_settings
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
    with tabs[7]:
        st.subheader("Algorithmic Strategy Tester (SMA Crossover)")
        
        c_strat1, c_strat2, c_strat3 = st.columns(3)
        short_window = c_strat1.number_input("Short MA", value=20, min_value=5)
        long_window = c_strat2.number_input("Long MA", value=50, min_value=10)
        initial_capital = c_strat3.number_input("Initial Capital", value=10000)
        
        if st.button("Run Backtest", type="primary"):
            data = market_data[active_ticker]['history'].copy()
            data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
            data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()
            data.dropna(inplace=True)
            
            data['Signal'] = 0
            data.loc[data['SMA_Short'] > data['SMA_Long'], 'Signal'] = 1
            data['Position'] = data['Signal'].diff()
            
            data['Market_Ret'] = data['Close'].pct_change()
            data['Strategy_Ret'] = data['Market_Ret'] * data['Signal'].shift(1)
            
            data['Equity_Strat'] = initial_capital * (1 + data['Strategy_Ret']).cumprod()
            data['Equity_BuyHold'] = initial_capital * (1 + data['Market_Ret']).cumprod()
            
            running_max = data['Equity_Strat'].cummax()
            data['Drawdown'] = (data['Equity_Strat'] / running_max) - 1
            
            total_ret_strat = (data['Equity_Strat'].iloc[-1] / initial_capital) - 1
            total_ret_bh = (data['Equity_BuyHold'].iloc[-1] / initial_capital) - 1
            
            st.divider()
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Strategy Return", f"{total_ret_strat:.2%}", delta=f"{total_ret_strat - total_ret_bh:.2%} vs Buy&Hold")
            kpi2.metric("Buy & Hold Return", f"{total_ret_bh:.2%}")
            kpi3.metric("Max Drawdown", f"{data['Drawdown'].min():.2%}")
            
            trade_count = data['Position'].abs().sum() / 2
            kpi4.metric("Total Trades", f"{int(trade_count)}")

            from plotly.subplots import make_subplots
            fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            
            fig_bt.add_trace(go.Scatter(x=data.index, y=data['Equity_Strat'], name='Strategy', line=dict(color='#00e676', width=2)), row=1, col=1)
            fig_bt.add_trace(go.Scatter(x=data.index, y=data['Equity_BuyHold'], name='Buy & Hold', line=dict(color='gray', dash='dash')), row=1, col=1)
            
            fig_bt.add_trace(go.Scatter(x=data.index, y=data['Drawdown'], name='Drawdown', fill='tozeroy', line=dict(color='#ef5350')), row=2, col=1)
            
            border_settings = dict(showline=True, linewidth=2, linecolor='rgba(255, 255, 255, 0.3)', mirror=True)
            fig_bt.update_layout(
                title="Equity Curve vs Benchmark",
                template="plotly_dark", height=500,
                xaxis=border_settings, yaxis=border_settings,
                xaxis2=border_settings, yaxis2=border_settings,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", y=1.02, x=0)
            )
            st.plotly_chart(fig_bt, use_container_width=True)
            
            st.subheader("Trade Log")
            trades = data[data['Position'] != 0].copy()
            if not trades.empty:
                trades['Action'] = trades['Position'].apply(lambda x: "BUY" if x == 1 else "SELL")
                trades['Price'] = trades['Close']
                trades['Value'] = trades['Equity_Strat']
                
                trade_log = trades[['Action', 'Price', 'Value']].style.format({
                    'Price': '${:.2f}', 
                    'Value': '${:.2f}'
                }).applymap(lambda x: 'color: #00e676' if x == 'BUY' else 'color: #ff1744', subset=['Action'])
                
                st.dataframe(trade_log, use_container_width=True)
            else:
                st.info("No trades generated with these parameters.")

    st.divider()
    
    footer_html = """
<style>
    .footer {
        text-align: center;
        padding-top: 20px;
        padding-bottom: 40px;
        color: #888;
        font-family: 'Source Sans Pro', sans-serif;
    }
    .footer-main {
        font-size: 16px;
        margin-bottom: 10px;
    }
    .footer-links a {
        color: #888;
        text-decoration: none;
        font-weight: 600;
        margin: 0 10px;
        transition: color 0.3s;
    }
    .footer-links a:hover {
        color: #e3e1e1;
        text-decoration: underline;
    }
    .footer-email {
        font-size: 14px;
        margin-top: 5px;
        color: #bbb;
    }
    .disclaimer {
        font-size: 11px;
        color: #555;
        margin-top: 25px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.4;
    }
</style>

<div class="footer">
    <div class="footer-main">
        Made with <span style="color: #ff1744;">&hearts;</span>
    </div>
    <div class="footer-links">
        <a href="mailto:rishi_ipm25@iift.edu">📩 Contact Me</a>
        <span style="color: #e3e1e1;">|</span>
        <a href="https://www.linkedin.com/in/rishi-dwivedi-433203227/">🔗 LinkedIn</a>
        <span style="color: #e3e1e1;">|</span>
        <a href="https://github.com/techrishidwivedi-wq">💻 GitHub</a>
    </div>
    <div class="disclaimer">
        <b>Disclaimer:</b> This application is for educational and research purposes only and does not constitute financial advice. 
        Trading financial markets involves significant risk. <br>
        Market data provided courtesy of Yahoo Finance API. 
        &copy; 2025 Trading Terminal. All Rights Reserved.
    </div>
</div>
"""
    
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
