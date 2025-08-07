import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from performance import MetricsCalculator

st.set_page_config(page_title="Pairs Trading Dashboard", layout="wide")
st.title("📈 Pairs Trading Live Dashboard")

# Sidebar controls
st.sidebar.header("Configuration")
data_path = st.sidebar.text_input("Path to backtest/results CSV", "results.csv")
lookback = st.sidebar.number_input("Z‑score lookback window", min_value=1, value=60)

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV with index column and at least:
      p1, p2, beta, signal, equity
    """
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    return df

# Load and preprocess
df = load_data(data_path)
# Compute spread and z-score if not already present
df["spread"] = df["p1"] - (df["beta"] * df["p2"])
rolling_mean = df["spread"].rolling(lookback).mean()
rolling_std  = df["spread"].rolling(lookback).std()
df["z_score"] = (df["spread"] - rolling_mean) / rolling_std

# Layout: three columns for plots
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Spread")
    fig1, ax1 = plt.subplots()
    ax1.plot(df.index, df["spread"], label="Spread")
    ax1.axhline(rolling_mean.mean(), linestyle="--", label="Mean Spread")
    ax1.set_ylabel("Price Difference")
    ax1.legend(loc="upper left")
    st.pyplot(fig1)

with col2:
    st.subheader("Z‑Score")
    fig2, ax2 = plt.subplots()
    ax2.plot(df.index, df["z_score"], label="Z‑Score")
    ax2.axhline( 2.0, color="red", linestyle="--", label="Entry Threshold")
    ax2.axhline(-2.0, color="red", linestyle="--")
    ax2.axhline( 0.5, color="green", linestyle="--", label="Exit Threshold")
    ax2.axhline(-0.5, color="green", linestyle="--")
    ax2.set_ylabel("Z‑Score")
    ax2.legend(loc="upper left")
    st.pyplot(fig2)

with col3:
    st.subheader("Equity Curve")
    fig3, ax3 = plt.subplots()
    ax3.plot(df.index, df["equity"], label="Equity")
    ax3.set_ylabel("Portfolio Value")
    ax3.legend(loc="upper left")
    st.pyplot(fig3)

#Performance metrics
st.markdown("---")
st.subheader("Performance Metrics")
returns = df["equity"].pct_change().fillna(0.0)
metrics = MetricsCalculator(returns)
st.table(metrics.report().to_frame(name="Value"))

#Raw data view
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(df)
