import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime

st.set_page_config(page_title="Mutual Fund Recommender Pro", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data/mutual_funds.csv', sep=';')
    # Convert numeric columns
    df["Net Asset Value (NAV)"] = pd.to_numeric(df["Net Asset Value (NAV)"], errors='coerce')
    df["1-Year Return (%)"] = pd.to_numeric(df["1-Year Return (%)"], errors='coerce') / 100  # convert to decimal
    df["3-Year Return (%)"] = pd.to_numeric(df["3-Year Return (%)"], errors='coerce') / 100
    df["5-Year Return (%)"] = pd.to_numeric(df["5-Year Return (%)"], errors='coerce') / 100
    return df

def calculate_lump_sum_maturity(principal, years, cagr):
    """Calculate maturity amount for lump sum investment."""
    return principal * (1 + cagr) ** years

def calculate_sip_maturity(monthly_investment, years, cagr):
    """Calculate maturity amount for SIP monthly investments."""
    r = cagr / 12
    n = years * 12
    maturity = monthly_investment * (((1 + r) ** n - 1) / r) * (1 + r)
    return maturity

def plot_growth_over_time(investment_mode, amount, years, cagr):
    """Return a plotly figure showing growth over years."""
    timeline = list(range(1, years + 1))
    values = []
    if investment_mode == "Lump Sum":
        for y in timeline:
            values.append(amount * (1 + cagr) ** y)
    else:  # SIP monthly
        monthly = amount
        r = cagr / 12
        for y in timeline:
            n = y * 12
            val = monthly * (((1 + r) ** n - 1) / r) * (1 + r)
            values.append(val)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline, y=values, mode='lines+markers', name='Investment Value'))
    fig.update_layout(
        title="Projected Investment Growth Over Time",
        xaxis_title="Years",
        yaxis_title="Value (INR)",
        template='plotly_white'
    )
    return fig

# Load data
df = load_data()

st.title("📊 Mutual Fund Recommender Pro")

# Sidebar Inputs
st.sidebar.header("Your Investment Details")

investment_mode = st.sidebar.selectbox("Investment Mode", ["Lump Sum", "SIP (Monthly)"])
investment_amount = st.sidebar.number_input("Investment Amount (INR)", min_value=1000, step=500, value=10000)
investment_duration = st.sidebar.slider("Investment Duration (Years)", min_value=1, max_value=30, value=5)
risk_preference = st.sidebar.selectbox("Risk Appetite", options=["Low", "Moderate", "High", "Any"], index=3)
category_filter = st.sidebar.multiselect("Mutual Fund Category", options=df["Category"].unique(), default=df["Category"].unique())

# Filter mutual funds by risk and category
filtered_df = df[
    (df["Risk Level"].str.lower().isin([risk_preference.lower()])) | (risk_preference == "Any")
]
filtered_df = filtered_df[filtered_df["Category"].isin(category_filter)]

st.subheader(f"Available Mutual Funds ({len(filtered_df)})")

# User selects fund(s)
selected_funds = st.multiselect("Select Mutual Fund(s) to Compare", options=filtered_df["Scheme Name"].tolist())

if not selected_funds:
    st.info("Please select at least one mutual fund to see recommendations and projections.")
    st.stop()

# Show table of selected funds
selected_df = filtered_df[filtered_df["Scheme Name"].isin(selected_funds)].copy()

# Choose CAGR basis: Use 3-Year Return if available else 1-Year Return
selected_df["CAGR"] = np.where(selected_df["3-Year Return (%)"] > 0, selected_df["3-Year Return (%)"], selected_df["1-Year Return (%)"])

# Calculate maturity amount for each selected fund
if investment_mode == "Lump Sum":
    selected_df["Maturity Amount (INR)"] = selected_df["CAGR"].apply(lambda cagr: calculate_lump_sum_maturity(investment_amount, investment_duration, cagr))
else:
    selected_df["Maturity Amount (INR)"] = selected_df["CAGR"].apply(lambda cagr: calculate_sip_maturity(investment_amount, investment_duration, cagr))

# Sort funds by maturity amount descending
selected_df = selected_df.sort_values(by="Maturity Amount (INR)", ascending=False)

st.dataframe(selected_df[[
    "Scheme Name", "Category", "Risk Level", "1-Year Return (%)", "3-Year Return (%)", "5-Year Return (%)", "Maturity Amount (INR)"
]].style.format({
    "1-Year Return (%)": "{:.2%}",
    "3-Year Return (%)": "{:.2%}",
    "5-Year Return (%)": "{:.2%}",
    "Maturity Amount (INR)": "₹ {:,.2f}"
}))

# Show graph for each selected fund's growth over time
st.subheader("Investment Growth Over Time")

for _, row in selected_df.iterrows():
    st.markdown(f"### {row['Scheme Name']} ({row['Category']}, Risk: {row['Risk Level']})")
    fig = plot_growth_over_time(investment_mode, investment_amount, investment_duration, row["CAGR"])
    st.plotly_chart(fig, use_container_width=True)

# Summary dashboard
st.sidebar.header("Summary")

st.sidebar.markdown(f"**Investment Mode:** {investment_mode}")
st.sidebar.markdown(f"**Investment Amount:** ₹{investment_amount:,.0f}")
st.sidebar.markdown(f"**Duration:** {investment_duration} years")
st.sidebar.markdown(f"**Selected Funds:** {len(selected_funds)}")

best_fund = selected_df.iloc[0]
st.sidebar.markdown(f"**Top Recommendation:** {best_fund['Scheme Name']}")
st.sidebar.markdown(f"Projected Maturity Amount: ₹{best_fund['Maturity Amount (INR)']:,.2f}")

st.markdown("---")
st.info("Maturity amount is an estimate based on historical CAGR returns and assumes consistent market conditions. Actual returns may vary.")

