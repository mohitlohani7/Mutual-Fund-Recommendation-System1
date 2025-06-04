import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/mutual_funds_enriched.csv", sep=';')
    df["Net Asset Value (NAV)"] = pd.to_numeric(df["Net Asset Value (NAV)"], errors='coerce')
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df.dropna(inplace=True)
    return df

# ------------------ Investment Calculator ------------------
def calculate_maturity(principal, years, rate):
    maturity = principal * ((1 + rate/100) ** years)
    return maturity

def compound_interest_table(principal, years, rate):
    data = []
    for year in range(1, years + 1):
        interest = principal * ((1 + rate/100) ** year)
        data.append({"Year": year, "Principal + Interest": round(interest, 2)})
    return pd.DataFrame(data)

# ------------------ Main App ------------------
st.set_page_config(page_title="Mutual Fund Advisor Dashboard", layout="wide")
st.title("📊 Mutual Fund Recommendation and Investment Calculator")

# Load Data
df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filter Funds")
risk_level = st.sidebar.selectbox("Select Risk Level", options=["Low", "Medium", "High"])

filtered_df = df[df["Risk Level"] == risk_level]

# Fund Selector
funds = filtered_df["Scheme Name"].unique()
selected_fund = st.selectbox("Choose a Mutual Fund Scheme", funds)
fund_df = filtered_df[filtered_df["Scheme Name"] == selected_fund].sort_values("Date")

# Fund Details
st.subheader("📈 Fund Performance Overview")
st.write(f"**Selected Scheme:** {selected_fund}")
st.write(f"**Risk Level:** {risk_level}")
st.write(f"**Latest NAV:** ₹{fund_df.iloc[-1]['Net Asset Value (NAV)']:.2f}")

# Line Chart - NAV over Time
fig = px.line(fund_df, x="Date", y="Net Asset Value (NAV)", title="NAV Over Time")
st.plotly_chart(fig, use_container_width=True)

# Pie Chart - Dummy Allocation (since real allocation not present)
st.subheader("💡 Asset Allocation (Dummy)")
st.plotly_chart(px.pie(values=[60, 25, 15], names=["Equity", "Debt", "Others"], title="Estimated Asset Allocation"))

# Investment Calculator
st.subheader("💰 Mutual Fund Maturity Calculator")
principal = st.number_input("Enter Investment Amount (₹)", min_value=1000, step=1000)
years = st.slider("Select Investment Duration (Years)", 1, 30, 5)
avg_rate = float(fund_df.iloc[-1]['5-Year Return (%)']) if years >= 5 else float(fund_df.iloc[-1]['3-Year Return (%)']) if years >= 3 else float(fund_df.iloc[-1]['1-Year Return (%)'])

maturity_amount = calculate_maturity(principal, years, avg_rate)
st.success(f"📢 Estimated Maturity Amount: ₹{maturity_amount:,.2f}")

# Maturity Bar Chart
bar_df = pd.DataFrame({"Amount Type": ["Principal", "Maturity"], "Value": [principal, maturity_amount]})
st.plotly_chart(px.bar(bar_df, x="Amount Type", y="Value", color="Amount Type", title="Principal vs Maturity"), use_container_width=True)

# Compound Table
st.subheader("📅 Year-wise Compound Interest Table")
compound_df = compound_interest_table(principal, years, avg_rate)
st.dataframe(compound_df)

# Recommendation Reasoning
st.info(f"✅ Based on your risk appetite (**{risk_level}**) and investment horizon (**{years} years**), the fund **{selected_fund}** is suitable due to an average return of **{avg_rate}%**.\n\nThis can help your investment grow steadily over time with calculated risk.")
