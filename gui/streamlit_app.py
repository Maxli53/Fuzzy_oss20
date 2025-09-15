# tradingview_style_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# ----------------------------
# Mock Data Generators
# ----------------------------
def generate_mock_ohlcv(symbol="AAPL", days=5, freq="1min"):
    """Generate mock OHLCV data for testing"""
    idx = pd.date_range(
        end=datetime.now(), periods=days * (390 if freq == "1min" else 78), freq=freq
    )
    price = np.cumsum(np.random.randn(len(idx))) + 150
    ohlcv = pd.DataFrame(index=idx)
    ohlcv["Open"] = price + np.random.randn(len(idx))
    ohlcv["High"] = ohlcv["Open"] + abs(np.random.randn(len(idx)))
    ohlcv["Low"] = ohlcv["Open"] - abs(np.random.randn(len(idx)))
    ohlcv["Close"] = ohlcv["Open"] + np.random.randn(len(idx)) * 0.5
    ohlcv["Volume"] = np.random.randint(1e4, 5e5, size=len(idx))
    ohlcv.reset_index(inplace=True)
    ohlcv.rename(columns={"index": "Datetime"}, inplace=True)
    return ohlcv

def mock_system_metrics():
    """Simulated system metrics"""
    return {
        "ticks_per_sec": np.random.randint(500, 2000),
        "latency_ms": round(np.random.uniform(10, 100), 2),
        "dropped_packets": np.random.randint(0, 5),
        "cache_hit_ratio": f"{np.random.uniform(80, 99):.2f}%",
    }

# ----------------------------
# Indicators
# ----------------------------
def add_rsi(df, period=14):
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -1 * delta.clip(upper=0).rolling(period).mean()
    rs = up / down
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, short=12, long=26, signal=9):
    df["EMA_short"] = df["Close"].ewm(span=short, adjust=False).mean()
    df["EMA_long"] = df["Close"].ewm(span=long, adjust=False).mean()
    df["MACD"] = df["EMA_short"] - df["EMA_long"]
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]
    return df

# ----------------------------
# Plotly Chart Builder
# ----------------------------
def plot_candles(df, symbol="AAPL"):
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df["Datetime"],
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Candles"
    ))

    # Volume
    fig.add_trace(go.Bar(
        x=df["Datetime"], y=df["Volume"],
        name="Volume", marker_color="rgba(128,128,128,0.5)", yaxis="y2"
    ))

    # Layout
    fig.update_layout(
        title=f"{symbol} - Candlestick Chart",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        height=600,
    )
    return fig

def plot_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["RSI"], mode="lines", name="RSI"))
    fig.add_hline(y=70, line=dict(color="red", dash="dot"))
    fig.add_hline(y=30, line=dict(color="green", dash="dot"))
    fig.update_layout(
        title="RSI Indicator", template="plotly_dark", height=200, margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

def plot_macd(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["Signal"], name="Signal"))
    fig.add_trace(go.Bar(x=df["Datetime"], y=df["Hist"], name="Histogram"))
    fig.update_layout(
        title="MACD Indicator", template="plotly_dark", height=200, margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(
    page_title="Pro Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Pro Trading Dashboard (TradingView-style)")
st.write("Mock demo dashboard with candlesticks, indicators, system monitor, and storage inspector")

# Sidebar controls
st.sidebar.header("🔧 Controls")
symbol = st.sidebar.text_input("Symbol", "AAPL")
timeframe = st.sidebar.selectbox("Timeframe", ["1min", "5min", "15min", "1h", "daily"])
days = st.sidebar.slider("Lookback Days", 1, 30, 5)
show_rsi = st.sidebar.checkbox("Show RSI", True)
show_macd = st.sidebar.checkbox("Show MACD", True)

# Generate mock data
df = generate_mock_ohlcv(symbol, days=days, freq="1min" if timeframe == "1min" else "5min")
df = add_rsi(df)
df = add_macd(df)

# Layout: main chart + indicators
main_col, side_col = st.columns([3, 1])

with main_col:
    st.plotly_chart(plot_candles(df, symbol), use_container_width=True)

    if show_rsi:
        st.plotly_chart(plot_rsi(df), use_container_width=True)

    if show_macd:
        st.plotly_chart(plot_macd(df), use_container_width=True)

with side_col:
    st.subheader("⚡ System Monitor")
    metrics = mock_system_metrics()
    st.metric("Ticks/sec", metrics["ticks_per_sec"])
    st.metric("Latency (ms)", metrics["latency_ms"])
    st.metric("Dropped Packets", metrics["dropped_packets"])
    st.metric("Cache Hit Ratio", metrics["cache_hit_ratio"])

    st.subheader("📂 Storage Inspector (Mock)")
    st.write("Available Symbols:")
    st.json({"AAPL": ["1m", "5m", "daily"], "MSFT": ["1m", "15m"], "EUR/USD": ["ticks", "1m"]})

    st.subheader("🕐 Sessions")
    st.write("- Pre-market: 04:00–09:30\n- RTH: 09:30–16:00\n- After-hours: 16:00–20:00")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: Mock generator")

