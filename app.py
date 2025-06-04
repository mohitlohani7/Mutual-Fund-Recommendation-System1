import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Mutual Fund Dashboard", layout="wide")

@st.cache_data

def load_data():
    df = pd.read_csv("data/mutual_funds_enriched.csv", sep=";")
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Net Asset Value"] = pd.to_numeric(df["Net Asset Value"], errors='coerce')
    df = df.dropna(subset=["Date", "Net Asset Value"])
    return df

df = load_data()

st.title("📊 Mutual Fund Investment Dashboard")

with st.sidebar:
    st.header("🧮 Investment Calculator")
    amount = st.number_input("💰 Monthly SIP Amount (₹)", min_value=500, step=500)
    years = st.slider("📆 Investment Duration (Years)", 1, 30, 10)
    expected_rate = st.slider("📈 Expected Annual Return (%)", 5.0, 20.0, 12.0)

    months = years * 12
    rate = expected_rate / 100 / 12

    maturity_value = amount * (((1 + rate)**months - 1) * (1 + rate)) / rate
    st.metric("Estimated Maturity Value", f"₹ {maturity_value:,.0f}")

    st.markdown("---")

fund_names = sorted(df["Scheme Name"].unique())
selected_fund = st.selectbox("Select a Mutual Fund", fund_names)
fund_df = df[df["Scheme Name"] == selected_fund].copy()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Net Asset Value Over Time")
    fund_df["Date"] = pd.to_datetime(fund_df["Date"], errors='coerce')
    fund_df = fund_df.dropna(subset=["Date", "Net Asset Value"])
    fund_df = fund_df.sort_values("Date")
    fig = px.line(fund_df, x="Date", y="Net Asset Value", title="NAV Over Time", markers=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Asset Allocation (Sample)")
    pie_data = pd.DataFrame({
        "Category": ["Equity", "Debt", "Cash", "Others"],
        "Allocation %": np.random.dirichlet(np.ones(4), size=1).flatten() * 100
    })
    fig2 = px.pie(pie_data, names="Category", values="Allocation %", title="Asset Allocation")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

st.subheader("📌 Fund Summary")

latest_nav = fund_df.sort_values("Date", ascending=False).iloc[0]

st.write(f"**Fund Name**: {selected_fund}")
st.write(f"**Latest NAV (₹)**: {latest_nav['Net Asset Value']:.2f} on {latest_nav['Date'].date()}")
st.write("**ISIN Growth:**", latest_nav["ISIN Div Payout/ ISIN Growth"])
st.write("**ISIN Reinvestment:**", latest_nav["ISIN Div Reinvestment"])

st.info("This dashboard provides estimated data. Please consult a financial advisor before investing.")
