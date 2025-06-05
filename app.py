import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Mutual Fund Recommender Pro", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/mutual_funds_enriched.csv", sep=';')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Net Asset Value (NAV)'] = pd.to_numeric(df['Net Asset Value (NAV)'], errors='coerce')
    
    # Add derived features
    df['Fund Age (years)'] = (pd.to_datetime('today') - df['Date']).dt.days / 365

    def get_category(name):
        name = name.lower()
        if "large cap" in name:
            return "Large Cap"
        elif "midcap" in name or "mid cap" in name:
            return "Mid Cap"
        elif "smallcap" in name or "small cap" in name:
            return "Small Cap"
        elif "tax" in name:
            return "Tax Saver"
        elif "hybrid" in name or "balanced" in name:
            return "Hybrid"
        else:
            return "Others"

    df['Category'] = df['Scheme Name'].apply(get_category)

    # Simulate AUM
    np.random.seed(42)
    df['AUM (Crores INR)'] = np.random.randint(100, 10000, size=len(df))

    # Clean & normalize returns
    df['1-Year Return (%)'] = pd.to_numeric(df['1-Year Return (%)'], errors='coerce')
    df['3-Year Return (%)'] = pd.to_numeric(df['3-Year Return (%)'], errors='coerce')
    df['5-Year Return (%)'] = pd.to_numeric(df['5-Year Return (%)'], errors='coerce')

    risk_map = {"Low": 1, "Moderate": 2, "High": 3}
    df['Risk Score'] = df['Risk Level'].map(risk_map)

    df['Momentum'] = ((df['1-Year Return (%)'] + df['3-Year Return (%)']) / 2) - df['5-Year Return (%)']

    # Normalize for scoring
    def normalize(col):
        return (col - col.min()) / (col.max() - col.min())

    df['Score'] = (
        normalize(df['3-Year Return (%)']) * 0.6 +
        normalize(df['AUM (Crores INR)']) * 0.2 +
        (1 - normalize(df['Risk Score'])) * 0.1 +
        normalize(df['Momentum']) * 0.1
    )

    return df.dropna()

df = load_data()

# Sidebar Inputs
st.sidebar.title("Investor Preferences")

risk = st.sidebar.selectbox("Risk Level", ['Low', 'Moderate', 'High'])
category = st.sidebar.multiselect("Fund Category", df['Category'].unique(), default=df['Category'].unique())
aum_threshold = st.sidebar.slider("Minimum AUM (₹ Cr)", 0, int(df['AUM (Crores INR)'].max()), 500)

inv_type = st.sidebar.radio("Investment Type", ['SIP', 'Lump Sum'])
years = st.sidebar.slider("Investment Duration (Years)", 1, 30, 5)

if inv_type == 'SIP':
    amount = st.sidebar.number_input("Monthly SIP Amount (₹)", min_value=500, step=500, value=5000)
else:
    amount = st.sidebar.number_input("Lump Sum Amount (₹)", min_value=1000, step=1000, value=50000)

# Filter data
filtered_df = df[
    (df['Risk Level'] == risk) &
    (df['Category'].isin(category)) &
    (df['AUM (Crores INR)'] >= aum_threshold)
]

if filtered_df.empty:
    st.warning("No funds match your criteria.")
    st.stop()

filtered_df = filtered_df.sort_values(by="Score", ascending=False)

# Top Funds Display
st.title("📈 Mutual Fund Recommender Pro")
st.subheader("Top 3 Mutual Funds for You")

for i, row in filtered_df.head(3).iterrows():
    st.markdown(f"### 🏦 {row['Scheme Name']}")
    st.write(f"- Risk Level: **{row['Risk Level']}**")
    st.write(f"- Category: **{row['Category']}**")
    st.write(f"- 3Y Return: **{row['3-Year Return (%)']:.2f}%**")
    st.write(f"- AUM: ₹{row['AUM (Crores INR)']:.2f} Cr")
    st.write(f"- Score: **{row['Score']:.3f}**")

# Investment Projection
top_return = filtered_df.iloc[0]['3-Year Return (%)'] / 100
if inv_type == 'Lump Sum':
    maturity = amount * ((1 + top_return) ** years)
else:
    monthly_rate = (1 + top_return) ** (1/12) - 1
    months = years * 12
    maturity = amount * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)

st.subheader("💹 Investment Projection")
st.write(f"Estimated Maturity Amount after {years} years: **₹{maturity:,.2f}**")

# Category Pie Chart
cat_data = filtered_df['Category'].value_counts()
st.plotly_chart(px.pie(names=cat_data.index, values=cat_data.values, title="Category Distribution"))

# Risk vs Return Bubble Chart
fig_bubble = px.scatter(
    filtered_df,
    x="Risk Score",
    y="3-Year Return (%)",
    size="AUM (Crores INR)",
    color="Category",
    hover_name="Scheme Name",
    title="Risk vs 3-Year Return Bubble Chart"
)
st.plotly_chart(fig_bubble, use_container_width=True)

# NAV Trend (Top Fund)
st.subheader("📉 NAV Trend of Top Fund")
nav_df = filtered_df[['Date', 'Net Asset Value (NAV)', 'Scheme Name']].sort_values('Date')
top_scheme = nav_df['Scheme Name'].iloc[0]
nav_df_top = nav_df[nav_df['Scheme Name'] == top_scheme]

if not nav_df_top.empty:
    fig_nav = px.line(nav_df_top, x='Date', y='Net Asset Value (NAV)', title=f"NAV Trend - {top_scheme}")
    st.plotly_chart(fig_nav, use_container_width=True)

st.markdown("---")
st.markdown("© 2025 Mutual Fund Recommender Pro | Powered by Streamlit")
