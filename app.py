import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Mutual Fund Recommendation System", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/mutual_funds_enriched.csv', sep=';')
    except FileNotFoundError:
        st.error("CSV file not found. Please make sure 'data/mutual_funds_enriched.csv' exists.")
        return pd.DataFrame()
    
    df["Net Asset Value (NAV)"] = pd.to_numeric(df["Net Asset Value (NAV)"], errors='coerce')
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df.dropna(subset=["Scheme Name", "Net Asset Value (NAV)", "Date", "Risk Level"], inplace=True)
    return df

def compound_interest(P, r, n, t):
    A = P * ((1 + r/n) ** (n * t))
    return round(A, 2)

# ----------------------- LOAD DATA -----------------------
df = load_data()

st.title("📈 Mutual Fund Recommendation Dashboard")

if df.empty:
    st.stop()

# ----------------------- USER INPUTS -----------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    amount = st.number_input("💰 Investment Amount (₹)", value=10000, step=1000)
with col2:
    tenure = st.slider("⏳ Investment Duration (Years)", 1, 30, 5)
with col3:
    mode = st.selectbox("💼 Investment Type", ["Lump Sum", "SIP"])
with col4:
    risk_profile = st.selectbox("⚠️ Risk Appetite", ["Low", "Moderate", "High"])

# ----------------------- FUND FILTERING -----------------------
filtered_df = df[df["Risk Level"].str.lower() == risk_profile.lower()]
if filtered_df.empty:
    st.warning("No mutual funds found matching your risk profile.")
    st.stop()

# ----------------------- FUND RECOMMENDATION -----------------------
best_fund = filtered_df.sort_values(by="5-Year Return (%)", ascending=False).iloc[0]

st.success(f"✅ Recommended Fund: **{best_fund['Scheme Name']}**")
st.markdown(f"- 5-Year Return: **{best_fund['5-Year Return (%)']}%**")
st.markdown(f"- Risk Level: **{best_fund['Risk Level']}**")

# ----------------------- MATURITY CALCULATION -----------------------
annual_rate = float(best_fund["5-Year Return (%)"]) / 100

if mode == "Lump Sum":
    maturity = compound_interest(amount, annual_rate, 1, tenure)
    st.info(f"📊 **Maturity Amount after {tenure} years** (Lump Sum): ₹{maturity}")
    principal = amount
else:
    # SIP formula: M = P × [((1 + r)^n - 1) / r] × (1 + r)
    monthly_rate = annual_rate / 12
    months = tenure * 12
    maturity = amount * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
    maturity = round(maturity, 2)
    principal = amount * months
    st.info(f"📊 **Maturity Amount after {tenure} years** (SIP): ₹{maturity}")

# ----------------------- YEAR-WISE TABLE -----------------------
st.markdown("### 📅 Year-wise Growth Table")
growth = []
for year in range(1, tenure + 1):
    if mode == "Lump Sum":
        value = compound_interest(amount, annual_rate, 1, year)
    else:
        value = amount * (((1 + monthly_rate) ** (year * 12) - 1) / monthly_rate) * (1 + monthly_rate)
        value = round(value, 2)
    growth.append({"Year": year, "Projected Value (₹)": value})
st.dataframe(pd.DataFrame(growth))

# ----------------------- PLOT NAV -----------------------
st.markdown("### 📈 NAV Over Time")
nav_data = df[df["Scheme Name"] == best_fund["Scheme Name"]].sort_values("Date")
if nav_data.empty:
    st.warning("No NAV data found. Displaying sample data.")
    from datetime import datetime, timedelta
    fake_dates = pd.date_range(end=datetime.today(), periods=10)
    fake_nav = np.linspace(100, 120, 10)
    nav_data = pd.DataFrame({
        "Date": fake_dates,
        "Net Asset Value (NAV)": fake_nav
    })
fig = px.line(nav_data, x="Date", y="Net Asset Value (NAV)", title="NAV Trend")
st.plotly_chart(fig, use_container_width=True)

# ----------------------- PIE CHART -----------------------
st.markdown("### 🥧 Fund Allocation (Sample)")
labels = ['Equity', 'Debt', 'Gold', 'Real Estate', 'Cash']
alloc = np.random.dirichlet(np.ones(5), size=1).flatten()
fig_pie = px.pie(values=alloc, names=labels, title='Estimated Fund Allocation')
st.plotly_chart(fig_pie, use_container_width=True)

# ----------------------- BAR COMPARISON -----------------------
st.markdown("### 📊 Investment Comparison")
bar_df = pd.DataFrame({
    "Type": ["Principal Invested", "Projected Maturity"],
    "Amount": [principal, maturity]
})
fig_bar = px.bar(bar_df, x="Type", y="Amount", color="Type", text="Amount", title="Investment vs Maturity")
st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------- END -----------------------
st.markdown("---")
st.caption("Made with ❤️ for interviews | Project by YOU")
