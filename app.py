import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# === Page Configuration ===
st.set_page_config(page_title="Mutual Fund Recommender Pro", layout="wide")

# === Load and Process Data ===
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('data/mutual_funds_enriched.csv', sep=';')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df["Net Asset Value (NAV)"] = pd.to_numeric(df["Net Asset Value (NAV)"], errors='coerce')

    # Fund age
    today = pd.to_datetime('today')
    df['Fund Age (years)'] = (today - df['Date']).dt.days / 365

    # Category
    def assign_category(name):
        name = name.lower()
        if 'large cap' in name:
            return 'Large Cap'
        elif 'midcap' in name or 'mid cap' in name:
            return 'Mid Cap'
        elif 'small cap' in name or 'smallcap' in name:
            return 'Small Cap'
        elif 'tax saver' in name or 'tax relief' in name or 'tax plan' in name:
            return 'Tax Saver'
        elif 'balanced' in name or 'hybrid' in name:
            return 'Hybrid'
        else:
            return 'Others'

    df['Category'] = df['Scheme Name'].apply(assign_category)

    # Add derived metrics
    np.random.seed(42)
    df['AUM (Crores INR)'] = np.random.randint(10, 10000, size=len(df))

    risk_map = {'Low': 1, 'Moderate': 2, 'High': 3}
    df['Risk Score'] = df['Risk Level'].map(risk_map).fillna(2)

    df['Momentum'] = ((df['1-Year Return (%)'] + df['3-Year Return (%)']) / 2) - df['5-Year Return (%)']

    def min_max_norm(series):
        return (series - series.min()) / (series.max() - series.min())

    df['Norm 3Y Return'] = min_max_norm(df['3-Year Return (%)'])
    df['Norm AUM'] = min_max_norm(df['AUM (Crores INR)'])
    df['Norm Risk'] = 1 - min_max_norm(df['Risk Score'])
    df['Norm Momentum'] = min_max_norm(df['Momentum'])

    df['Score'] = (
        df['Norm 3Y Return'] * 0.7 +
        df['Norm AUM'] * 0.15 +
        df['Norm Risk'] * 0.10 +
        df['Norm Momentum'] * 0.05
    )

    return df

# === Load Data ===
df = load_and_process_data()

# === UI Header ===
st.title("💼 Mutual Fund Recommender Pro")

# === Sidebar Inputs ===
st.sidebar.header("Customize Your Investment")

risk_input = st.sidebar.selectbox("Select Risk Appetite", ['Low', 'Moderate', 'High'])
category_input = st.sidebar.multiselect("Select Fund Category", options=df['Category'].unique(), default=list(df['Category'].unique()))
min_aum_input = st.sidebar.slider("Minimum AUM (in Crores)", 0, int(df['AUM (Crores INR)'].max()), 100)

inv_type = st.sidebar.radio("Investment Type", ['Lump Sum', 'SIP'])

if inv_type == 'Lump Sum':
    lump_sum_amount = st.sidebar.number_input("Lump Sum Amount (₹)", min_value=1000, step=1000, value=50000)
    inv_period = st.sidebar.slider("Investment Period (Years)", 1, 20, 5)
else:
    sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", min_value=500, step=500, value=5000)
    inv_period = st.sidebar.slider("Investment Period (Years)", 1, 20, 5)

# === Filter Data ===
filtered_df = df[
    (df['Risk Level'] == risk_input) &
    (df['Category'].isin(category_input)) &
    (df['AUM (Crores INR)'] >= min_aum_input)
]

if filtered_df.empty:
    st.warning("No funds found for your selected criteria.")
    st.stop()

filtered_df = filtered_df.sort_values(by='Score', ascending=False)

# === Recommendation Section ===
st.subheader("🌟 Top Recommended Funds")
for idx, row in filtered_df.head(3).iterrows():
    st.markdown(f"### {row['Scheme Name']} (Score: {row['Score']:.3f})")
    st.write(f"- **Category**: {row['Category']}")
    st.write(f"- **Risk Level**: {row['Risk Level']}")
    st.write(f"- **AUM**: ₹{row['AUM (Crores INR)']:.2f} Cr")
    st.write(f"- **3-Year Return**: {row['3-Year Return (%)']:.2f}%")
    st.write(f"- **Momentum**: {row['Momentum']:.2f}")
    st.markdown("---")

# === Bubble Chart ===
st.subheader("📊 Risk vs Return Visualization")
fig = px.scatter(
    filtered_df,
    x='Risk Score',
    y='3-Year Return (%)',
    size='AUM (Crores INR)',
    color='Category',
    hover_name='Scheme Name',
    title='Risk vs 3-Year Return (Bubble size = AUM)'
)
st.plotly_chart(fig, use_container_width=True)

# === Investment Calculator ===
st.header("💰 Investment Growth Estimator")

if inv_type == 'Lump Sum':
    annual_return = filtered_df.iloc[0]['3-Year Return (%)'] / 100
    principal = lump_sum_amount
    maturity = principal * ((1 + annual_return) ** inv_period)

    st.markdown(f"**Lump Sum Investment:** ₹{principal:,}")
    st.markdown(f"**Expected Value after {inv_period} years:** ₹{maturity:,.2f}")

    years = list(range(inv_period + 1))
    values = [principal * ((1 + annual_return) ** y) for y in years]

    df_growth = pd.DataFrame({
        "Year": years,
        "Investment Value": values,
        "Principal": [principal] * len(values)
    })

    fig2 = px.line(df_growth, x='Year', y=['Principal', 'Investment Value'], markers=True, title="Lump Sum Growth Over Time")
    st.plotly_chart(fig2, use_container_width=True)

else:
    months = inv_period * 12
    annual_return = filtered_df.iloc[0]['3-Year Return (%)'] / 100
    monthly_return = (1 + annual_return) ** (1 / 12) - 1

    fv = sip_amount * (((1 + monthly_return) ** months - 1) / monthly_return) * (1 + monthly_return)

    st.markdown(f"**Monthly SIP:** ₹{sip_amount:,}")
    st.markdown(f"**Expected Value after {inv_period} years:** ₹{fv:,.2f}")

    years = list(range(inv_period + 1))
    sip_growth = []
    for y in years:
        m = y * 12
        if m == 0:
            sip_growth.append(0)
        else:
            sip_growth.append(sip_amount * (((1 + monthly_return) ** m - 1) / monthly_return) * (1 + monthly_return))

    df_sip = pd.DataFrame({
        "Year": years,
        "Investment Value": sip_growth,
        "Principal": [sip_amount * 12 * y for y in years]
    })

    fig3 = px.line(df_sip, x='Year', y=['Principal', 'Investment Value'], markers=True, title="SIP Growth Over Time")
    st.plotly_chart(fig3, use_container_width=True)

# === Pie Chart for Fund Distribution ===
st.header("📈 Fund Category Distribution")
cat_counts = filtered_df['Category'].value_counts()
fig4 = px.pie(values=cat_counts.values, names=cat_counts.index, title="Selected Fund Categories")
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.caption("© 2025 Mutual Fund Recommender Pro — For Educational Use Only")
