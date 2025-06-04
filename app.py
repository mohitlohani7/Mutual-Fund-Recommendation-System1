import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv('data/mutual_funds_enriched.csv', sep=';')
    df["NAV"] = pd.to_numeric(df["NAV"], errors='coerce')
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["NAV", "Date"])
    return df

def maturity_amount_lumpsum(principal, rate, years):
    return principal * ((1 + rate/100) ** years)

def maturity_amount_sip(monthly, rate, years):
    n = years * 12
    r = rate / (12 * 100)
    return monthly * (((1 + r) ** n - 1) * (1 + r)) / r

# Load data
df = load_data()
st.set_page_config(page_title="Mutual Fund Explorer", layout="wide")

st.title("📊 Mutual Fund Investment Dashboard")

# Sidebar Filters
st.sidebar.header("🔍 Filter Funds")
fund_names = df["Scheme Name"].unique()
selected_fund = st.sidebar.selectbox("Choose a Mutual Fund", fund_names)

risk_levels = df["Risk"].unique().tolist()
selected_risk = st.sidebar.multiselect("Select Risk Level", risk_levels, default=risk_levels)

filtered_df = df[(df["Scheme Name"] == selected_fund) & (df["Risk"].isin(selected_risk))]

# Line Graph - NAV over time
st.subheader(f"NAV Over Time - {selected_fund}")
if not filtered_df.empty:
    fig = px.line(filtered_df.sort_values("Date"), x="Date", y="NAV", title=f"{selected_fund} NAV Trend")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for selected filters.")

# Pie C
