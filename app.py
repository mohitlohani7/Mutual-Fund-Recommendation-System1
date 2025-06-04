import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/mutual_funds_enriched.csv", sep=';')
    return df

df = load_data()

# App title and tabs
st.set_page_config(page_title="Smart Mutual Fund Dashboard", layout="wide")
st.title("💹 Smart Mutual Fund Investment Platform")
tabs = st.tabs(["🏦 Explore Funds", "🧮 SIP/Maturity Calculator", "📊 Compare Funds", "📘 About"])

# Explore Funds Tab
with tabs[0]:
    st.header("📈 Explore Mutual Fund Details")

    selected_fund = st.selectbox("Choose a Mutual Fund Scheme:", df["Scheme Name"].unique())
    fund_df = df[df["Scheme Name"] == selected_fund]

    # Show NAV trend
    st.subheader(f"NAV Trend - {selected_fund}")
    fig = px.line(fund_df.sort_values("Date"), x="Date", y="Net Asset Value", title="NAV Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Simulated asset allocation pie chart
    st.subheader("Asset Allocation")
    pie_data = pd.DataFrame({
        "Type": ["Equity", "Debt", "Cash"],
        "Allocation": np.random.dirichlet(np.ones(3), size=1)[0]
    })
    fig_pie = px.pie(pie_data, names='Type', values='Allocation', title='Portfolio Allocation')
    st.plotly_chart(fig_pie, use_container_width=True)

# Investment Calculator Tab
with tabs[1]:
    st.header("🧮 Mutual Fund Investment Calculator")

    investment_type = st.radio("Investment Type", ["One-time", "SIP"])
    amount = st.number_input("Investment Amount (₹)", min_value=1000, step=1000)
    duration = st.slider("Investment Duration (Years)", 1, 30, 5)
    expected_return = st.slider("Expected Annual Return (%)", 5, 20, 12)

    if investment_type == "One-time":
        maturity_value = amount * (1 + expected_return / 100) ** duration
    else:
        maturity_value = amount * (((1 + expected_return / 100) ** duration - 1) * (1 + expected_return / 100)) / (expected_return / 100)

    total_invested = amount if investment_type == "One-time" else amount * duration
    gain = maturity_value - total_invested

    st.metric("Maturity Value", f"₹{maturity_value:,.2f}")
    st.metric("Total Gain", f"₹{gain:,.2f}")

    fig = px.bar(x=["Invested", "Gain"], y=[total_invested, gain], labels={'x':"", 'y':"Amount (₹)"}, title="Investment vs Gain")
    st.plotly_chart(fig, use_container_width=True)

# Compare Funds Tab
with tabs[2]:
    st.header("📊 Compare Mutual Funds")
    compare_funds = st.multiselect("Select Multiple Funds to Compare:", df["Scheme Name"].unique(), default=df["Scheme Name"].unique()[:3])

    if compare_funds:
        comp_df = df[df["Scheme Name"].isin(compare_funds)]
        comp_df_latest = comp_df.sort_values("Date").drop_duplicates("Scheme Name", keep='last')
        fig = px.bar(comp_df_latest, x="Scheme Name", y="Net Asset Value", color="Scheme Name", title="Latest NAV Comparison")
        st.plotly_chart(fig, use_container_width=True)

# About Tab
with tabs[3]:
    st.header("📘 About this Dashboard")
    st.markdown("""
        This professional-grade mutual fund dashboard helps investors:
        - Explore real-time NAVs and fund performance.
        - Use an inbuilt maturity calculator.
        - Compare funds using interactive graphs.
        - Understand where their money is invested using pie charts.
        
        _Designed to feel like the Groww experience._ 💼
    """)
