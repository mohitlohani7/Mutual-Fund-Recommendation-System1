import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Mutual Fund Dashboard", layout="wide")
st.title("📊 Mutual Fund Investment Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/mutual_funds_enriched.csv", sep=';')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Net Asset Value (NAV)'] = pd.to_numeric(df['Net Asset Value (NAV)'], errors='coerce')
    df = df.dropna(subset=['Scheme Name', 'Date', 'Net Asset Value (NAV)'])
    return df

df = load_data()

# Sidebar: Fund selection
with st.sidebar:
    st.header("🔍 Filter Funds")
    selected_fund = st.selectbox("Select a Mutual Fund Scheme", df['Scheme Name'].unique())
    investment_amount = st.number_input("Investment Amount (₹)", min_value=1000, step=500)
    years = st.slider("Investment Duration (Years)", 1, 30, 5)
    risk_level = st.selectbox("Your Risk Appetite", ["Low", "Moderate", "High"])

# Filtered data
fund_df = df[df['Scheme Name'] == selected_fund].sort_values("Date")

# Show latest NAV
st.subheader(f"📌 Fund Details: {selected_fund}")
if not fund_df.empty:
    st.write(f"**Latest NAV:** ₹{fund_df.iloc[-1]['Net Asset Value (NAV)']:.2f}")
    
    # NAV over time graph
    fig = px.line(fund_df, x="Date", y="Net Asset Value (NAV)", title="Net Asset Value Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Calculate maturity using 1Y return (approximation)
    try:
        rate = fund_df.iloc[-1]['1-Year Return (%)'] / 100
        compound_maturity = investment_amount * ((1 + rate) ** years)

        st.markdown("### 📈 Maturity Calculation")
        st.success(f"Estimated Value after {years} years: ₹{compound_maturity:,.2f}")

        # Show breakdown table
        st.markdown("#### 📅 Year-wise Growth Table")
        growth_data = pd.DataFrame({
            "Year": list(range(1, years + 1)),
            "Principal (₹)": investment_amount,
            "Interest Rate (%)": round(rate * 100, 2),
            "Maturity Value (₹)": [investment_amount * ((1 + rate) ** i) for i in range(1, years + 1)]
        })
        st.dataframe(growth_data, use_container_width=True)

        # Graph: Principal vs Growth
        st.markdown("#### 📊 Principal vs Maturity Graph")
        chart_df = pd.DataFrame({
            "Year": list(range(1, years + 1)),
            "Principal": [investment_amount] * years,
            "Estimated Value": [investment_amount * ((1 + rate) ** i) for i in range(1, years + 1)]
        })
        fig_area = px.area(chart_df, x="Year", y=["Principal", "Estimated Value"], 
                          title="Investment Growth Over Time")
        st.plotly_chart(fig_area, use_container_width=True)

        # Pie chart: hypothetical asset allocation
        st.markdown("#### 🥧 Hypothetical Asset Allocation")
        pie = px.pie(names=["Equity", "Debt", "Others"], values=[60, 30, 10],
                     title="Portfolio Distribution")
        st.plotly_chart(pie, use_container_width=True)

        # Recommendation block
        st.markdown("### 🤖 Fund Recommendation")
        risk_map = {"Low": ["Low"], "Moderate": ["Low", "Moderate"], "High": ["Low", "Moderate", "High"]}
        if fund_df.iloc[-1]['Risk Level'] in risk_map[risk_level]:
            st.success("✅ This fund matches your risk profile.")
        else:
            st.warning("⚠️ This fund may not match your preferred risk level.")

        st.info(f"Why invest in this fund?\n\n- Consistent returns of {fund_df.iloc[-1]['1-Year Return (%)']}% last year\n- Suitable for {fund_df.iloc[-1]['Risk Level']} risk investors\n- Long-term compounding benefits")

    except Exception as e:
        st.error("⚠️ Calculation error. Please check the data format.")
else:
    st.warning("No data found for the selected mutual fund.")
