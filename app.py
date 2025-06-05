import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Mutual Fund Recommender Pro", layout="wide")

@st.cache_data
def load_and_process_data():
    df = pd.read_csv('data/mutual_funds_enriched.csv', sep=';')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df["Net Asset Value (NAV)"] = pd.to_numeric(df["Net Asset Value (NAV)"], errors='coerce')
    
    today = pd.to_datetime('today')
    df['Fund Age (years)'] = (today - df['Date']).dt.days / 365

    def assign_category(name):
        name = str(name).lower()
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

    # Fill missing returns with median values
    df['1-Year Return (%)'] = pd.to_numeric(df['1-Year Return (%)'], errors='coerce').fillna(df['1-Year Return (%)'].median())
    df['3-Year Return (%)'] = pd.to_numeric(df['3-Year Return (%)'], errors='coerce').fillna(df['3-Year Return (%)'].median())
    df['5-Year Return (%)'] = pd.to_numeric(df['5-Year Return (%)'], errors='coerce').fillna(df['5-Year Return (%)'].median())

    # Random AUM for demo
    np.random.seed(42)
    df['AUM (Crores INR)'] = np.random.randint(10, 10000, size=len(df))

    risk_map = {'Low': 1, 'Moderate': 2, 'High': 3}
    df['Risk Score'] = df['Risk Level'].map(risk_map).fillna(2)

    df['Momentum'] = ((df['1-Year Return (%)'] + df['3-Year Return (%)']) / 2) - df['5-Year Return (%)']

    def min_max_norm(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-6)

    df['Norm 3Y Return'] = min_max_norm(df['3-Year Return (%)'])
    df['Norm AUM'] = min_max_norm(df['AUM (Crores INR)'])
    df['Norm Risk'] = 1 - min_max_norm(df['Risk Score'])  # Lower risk gets higher score
    df['Norm Momentum'] = min_max_norm(df['Momentum'])

    df['Score'] = (
        df['Norm 3Y Return'] * 0.7 +
        df['Norm AUM'] * 0.15 +
        df['Norm Risk'] * 0.10 +
        df['Norm Momentum'] * 0.05
    )
    return df

# Load data
df = load_and_process_data()

# === Streamlit UI ===
st.title("🚀 Mutual Fund Recommender Pro")

risk_input = st.sidebar.selectbox("Select your Risk Appetite", ['Low', 'Moderate', 'High'])
category_input = st.sidebar.multiselect("Select Fund Category", options=df['Category'].unique(), default=list(df['Category'].unique()))
min_aum_input = st.sidebar.slider("Minimum AUM (Crores INR)", 0, int(df['AUM (Crores INR)'].max()), 100)

inv_type = st.sidebar.radio("Investment Type", ['Lump Sum', 'SIP'])

if inv_type == 'Lump Sum':
    lump_sum_amount = st.sidebar.number_input("Lump Sum Amount (₹)", min_value=1000, step=1000, value=50000)
    inv_period = st.sidebar.slider("Investment Period (Years)", 1, 20, 5)
else:
    sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", min_value=500, step=500, value=5000)
    inv_period = st.sidebar.slider("Investment Period (Years)", 1, 20, 5)

# Filter funds
filtered_df = df[
    (df['Risk Level'] == risk_input) &
    (df['Category'].isin(category_input)) &
    (df['AUM (Crores INR)'] >= min_aum_input)
]

if filtered_df.empty:
    st.warning("🚫 No mutual funds match your selected filters.")
    st.markdown("Please broaden your filters from the sidebar.")
else:
    filtered_df = filtered_df.sort_values(by='Score', ascending=False)

    st.subheader("🎯 Top 3 Recommended Funds")
    for idx, row in filtered_df.head(3).iterrows():
        st.markdown(f"### {row['Scheme Name']}  (Score: {row['Score']:.3f})")
        st.write(f"- **Category:** {row['Category']}")
        st.write(f"- **Risk Level:** {row['Risk Level']}")
        st.write(f"- **AUM:** ₹{row['AUM (Crores INR)']:.2f} Cr")
        st.write(f"- **3Y Return:** {row['3-Year Return (%)']:.2f}%")
        st.write(f"- **Momentum:** {row['Momentum']:.2f}")
        st.success("📌 Recommended for strong 3Y return, consistent AUM, and balanced risk.")

    fig = px.scatter(
        filtered_df,
        x='Risk Score',
        y='3-Year Return (%)',
        size='AUM (Crores INR)',
        color='Category',
        hover_name='Scheme Name',
        title='Risk vs 3-Year Return (Bubble = AUM)',
        labels={'Risk Score': 'Risk (1=Low, 3=High)', '3-Year Return (%)': '3-Year Return (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.header("📈 Investment Calculator")
    if inv_type == 'Lump Sum':
        annual_return = filtered_df.iloc[0]['3-Year Return (%)'] / 100
        principal = lump_sum_amount
        years = inv_period
        maturity_value = principal * ((1 + annual_return) ** years)

        st.write(f"### Invested: ₹{principal}")
        st.write(f"### Projected Value in {years} years: ₹{maturity_value:,.2f}")

        timeline = list(range(years + 1))
        values = [principal * ((1 + annual_return) ** y) for y in timeline]

        df_growth = pd.DataFrame({
            "Year": timeline,
            "Investment Value": values,
            "Principal": [principal] * (years + 1)
        })

        fig2 = px.line(df_growth, x='Year', y=['Principal', 'Investment Value'], title='Lump Sum Growth Over Time', markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    else:
        months = inv_period * 12
        annual_return = filtered_df.iloc[0]['3-Year Return (%)'] / 100
        monthly_return = (1 + annual_return) ** (1/12) - 1
        sip = sip_amount

        fv = sip * (((1 + monthly_return) ** months - 1) / monthly_return) * (1 + monthly_return)

        st.write(f"### SIP Amount: ₹{sip} / month")
        st.write(f"### Estimated Maturity after {inv_period} years: ₹{fv:,.2f}")

        timeline = list(range(inv_period + 1))
        fv_list = []
        for y in timeline:
            m = y * 12
            if m == 0:
                fv_list.append(0)
            else:
                fv_list.append(sip * (((1 + monthly_return) ** m - 1) / monthly_return) * (1 + monthly_return))

        df_sip_growth = pd.DataFrame({
            "Year": timeline,
            "Investment Value": fv_list,
            "Principal": [sip * 12 * y for y in timeline]
        })

        fig3 = px.line(df_sip_growth, x='Year', y=['Principal', 'Investment Value'], title='SIP Growth Over Time', markers=True)
        st.plotly_chart(fig3, use_container_width=True)

    st.header("📊 Fund Category Distribution")
    cat_counts = filtered_df['Category'].value_counts()
    fig_pie = px.pie(values=cat_counts.values, names=cat_counts.index, title='Fund Category Breakdown')
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("© 2025 Mutual Fund Recommender Pro | Educational Demo App")
