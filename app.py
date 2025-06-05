import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime

st.set_page_config(page_title="Mutual Fund Recommender Pro", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/mutual_funds_enriched.csv", sep=';')
    # Clean column names if needed
    df.columns = df.columns.str.strip()
    # Convert returns to numeric
    df["1-Year Return (%)"] = pd.to_numeric(df["1-Year Return (%)"], errors='coerce')
    df["3-Year Return (%)"] = pd.to_numeric(df["3-Year Return (%)"], errors='coerce')
    df["5-Year Return (%)"] = pd.to_numeric(df["5-Year Return (%)"], errors='coerce')
    return df

df = load_data()

st.title("📊 Mutual Fund Investment Recommender")

# Sidebar Inputs
st.sidebar.header("Investment Details")

investment_type = st.sidebar.selectbox("Investment Type", ["Lump Sum", "SIP"])

amount = st.sidebar.number_input("Investment Amount (₹)", min_value=1000, step=1000)

investment_duration_years = st.sidebar.slider("Investment Duration (Years)", min_value=1, max_value=30, value=5)

risk_levels = df["Risk Level"].unique().tolist()
risk_level = st.sidebar.selectbox("Risk Level", risk_levels)

categories = df["Category"].unique().tolist()
category = st.sidebar.selectbox("Mutual Fund Category", categories)

# Filter dataframe by risk level and category
filtered_df = df[(df["Risk Level"] == risk_level) & (df["Category"] == category)]

if filtered_df.empty:
    st.warning("No mutual funds found for the selected Risk Level and Category.")
    st.stop()

st.subheader(f"Available Mutual Funds for Risk: {risk_level} & Category: {category}")

# Show filtered funds summary
st.dataframe(filtered_df[["Scheme Name", "Net Asset Value (NAV)", "1-Year Return (%)", "3-Year Return (%)", "5-Year Return (%)", "Risk Level"]].reset_index(drop=True))

# Function to calculate maturity amount for lump sum
def calc_lump_sum_maturity(P, r, t):
    # P = Principal invested, r = annual return rate in decimal, t = time in years
    A = P * ((1 + r) ** t)
    return A

# Function to calculate maturity amount for SIP (monthly)
def calc_sip_maturity(monthly_investment, r, t):
    # monthly_investment = amount invested per month
    # r = annual rate of return in decimal
    # t = time in years
    n = t * 12  # total months
    monthly_rate = (1 + r) ** (1/12) - 1
    A = monthly_investment * (( (1 + monthly_rate) ** n - 1) / monthly_rate) * (1 + monthly_rate)
    return A

# Choose which return to use for CAGR estimation
st.sidebar.markdown("### Choose Return Duration to Calculate Estimated Returns")
return_duration = st.sidebar.radio("Return Duration", ["1-Year Return (%)", "3-Year Return (%)", "5-Year Return (%)"])

# Calculate estimated maturity for each fund
estimates = []
for idx, row in filtered_df.iterrows():
    r = row[return_duration] / 100  # convert % to decimal
    if investment_type == "Lump Sum":
        maturity = calc_lump_sum_maturity(amount, r, investment_duration_years)
    else:
        maturity = calc_sip_maturity(amount, r, investment_duration_years)
    estimates.append(maturity)

filtered_df["Estimated Maturity (₹)"] = estimates

# Sort funds by estimated maturity descending
filtered_df = filtered_df.sort_values(by="Estimated Maturity (₹)", ascending=False).reset_index(drop=True)

st.subheader("Recommended Mutual Funds Based on Your Inputs")

# Display top 10 recommendations
top_n = 10
top_funds = filtered_df.head(top_n)
st.dataframe(top_funds[["Scheme Name", "Risk Level", "Category", return_duration, "Estimated Maturity (₹)"]])

# Plot maturity amount bar chart
fig = go.Figure()
fig.add_trace(go.Bar(
    x=top_funds["Scheme Name"],
    y=top_funds["Estimated Maturity (₹)"],
    marker_color='indianred'
))
fig.update_layout(
    title=f"Estimated Maturity Amount after {investment_duration_years} Years",
    xaxis_title="Mutual Fund Scheme",
    yaxis_title="Maturity Amount (₹)",
    xaxis_tickangle=-45,
    height=500,
    margin=dict(t=50, b=150)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
---
**Note:**  
- The returns are estimated based on historical annual returns; actual returns may vary.  
- SIP assumes monthly investments on a compounding basis.  
- Lump sum assumes one-time investment compounding annually.  
- Always consult a financial advisor before investing.
""")
