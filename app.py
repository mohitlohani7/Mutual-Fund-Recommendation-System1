import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Mutual Fund Investment Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data/mutual_funds_enriched.csv', sep=';')
    # Clean & type cast
    df["Net Asset Value (NAV)"] = pd.to_numeric(df["Net Asset Value (NAV)"], errors='coerce')
    df["1-Year Return (%)"] = pd.to_numeric(df["1-Year Return (%)"], errors='coerce')
    df["3-Year Return (%)"] = pd.to_numeric(df["3-Year Return (%)"], errors='coerce')
    df["5-Year Return (%)"] = pd.to_numeric(df["5-Year Return (%)"], errors='coerce')
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    return df.dropna(subset=["Net Asset Value (NAV)"])

df = load_data()

st.title("📊 Mutual Fund Investment Dashboard & Calculator")
st.markdown("""
Welcome! Select your risk appetite, choose a mutual fund, and estimate your investment maturity amount with SIP or Lump Sum options.
""")

# --- Step 1: User Risk Appetite ---
risk_profile = st.selectbox("Select Your Risk Appetite:", options=["Low", "Moderate", "High"])

# Filter funds by risk
filtered_funds = df[df["Risk Level"].str.lower() == risk_profile.lower()]
if filtered_funds.empty:
    st.warning("No mutual funds found for selected risk profile. Please try another risk level.")
    st.stop()

# --- Step 2: Select Fund ---
selected_fund_name = st.selectbox("Select Mutual Fund:", filtered_funds["Scheme Name"].unique())
fund_df = filtered_funds[filtered_funds["Scheme Name"] == selected_fund_name].sort_values("Date")

st.markdown(f"### Fund Details for **{selected_fund_name}**")
latest_nav = fund_df.iloc[-1]["Net Asset Value (NAV)"]
st.write(f"**Latest NAV:** ₹{latest_nav:.2f}")
st.write(f"**Risk Level:** {risk_profile}")
st.write(f"**5-Year Return:** {fund_df.iloc[-1]['5-Year Return (%)']:.2f}%")

# --- Step 3: Investment Type ---
investment_type = st.radio("Choose Investment Type:", ["Lump Sum", "SIP"])

# --- Step 4: Investment Inputs ---
col1, col2, col3 = st.columns(3)
with col1:
    principal = st.number_input("Investment Amount (₹)", min_value=1000, step=1000, value=50000)
with col2:
    duration_years = st.slider("Investment Duration (years)", 1, 30, 5)
with col3:
    if investment_type == "SIP":
        monthly_investment = st.number_input("Monthly SIP Amount (₹)", min_value=1000, step=1000, value=5000)
    else:
        monthly_investment = None

# --- Step 5: Calculate Maturity ---

# Use 5-Year Return (%) as approx CAGR for calculation
cagr = fund_df.iloc[-1]["5-Year Return (%)"] / 100

def compound_interest(principal, rate, time):
    return principal * (1 + rate) ** time

def calculate_sip_maturity(monthly_amount, rate, years):
    # Monthly rate approx
    r = rate / 12
    n = years * 12
    maturity = monthly_amount * (( (1 + r) ** n - 1) / r) * (1 + r)
    return maturity

if investment_type == "Lump Sum":
    maturity_amount = compound_interest(principal, cagr, duration_years)
    total_invested = principal
else:
    maturity_amount = calculate_sip_maturity(monthly_investment, cagr, duration_years)
    total_invested = monthly_investment * 12 * duration_years

interest_earned = maturity_amount - total_invested

# --- Step 6: Show Results ---
st.subheader("Investment Summary")
st.write(f"**Total Principal Invested:** ₹{total_invested:,.2f}")
st.write(f"**Estimated Maturity Amount:** ₹{maturity_amount:,.2f}")
st.write(f"**Interest Earned:** ₹{interest_earned:,.2f}")
st.write(f"**Approximate CAGR used for calculation:** {cagr*100:.2f}%")

# --- Step 7: Visualizations ---

# Principal vs Maturity bar chart
summary_df = pd.DataFrame({
    "Amount": [total_invested, maturity_amount],
    "Type": ["Principal Invested", "Maturity Amount"]
})

fig_bar = px.bar(summary_df, x="Type", y="Amount", color="Type", text="Amount", title="Investment vs Maturity")
fig_bar.update_traces(texttemplate='₹%{text:.2s}', textposition='outside')
st.plotly_chart(fig_bar, use_container_width=True)

# Pie chart for Asset Allocation (dummy example)
# You can replace with real data if you have sector allocation info
st.subheader("Sample Asset Allocation in This Mutual Fund")
asset_alloc = {
    "Equity": 65,
    "Debt": 25,
    "Cash & Others": 10
}
fig_pie = px.pie(
    names=list(asset_alloc.keys()),
    values=list(asset_alloc.values()),
    title="Asset Allocation",
    hole=0.4,
    color_discrete_sequence=px.colors.sequential.RdBu
)
st.plotly_chart(fig_pie, use_container_width=True)

# --- Step 8: Year-wise breakdown table ---

st.subheader("Year-wise Investment Growth")

if investment_type == "Lump Sum":
    years = list(range(duration_years + 1))
    principal_over_years = [principal] * len(years)
    maturity_over_years = [compound_interest(principal, cagr, y) for y in years]
    interest_over_years = [maturity_over_years[i] - principal_over_years[i] for i in range(len(years))]
else:
    years = list(range(duration_years + 1))
    principal_over_years = [monthly_investment * 12 * y for y in years]
    maturity_over_years = [calculate_sip_maturity(monthly_investment, cagr, y) for y in years]
    interest_over_years = [maturity_over_years[i] - principal_over_years[i] for i in range(len(years))]

table_data = pd.DataFrame({
    "Year": years,
    "Principal Invested (₹)": [f"{x:,.2f}" for x in principal_over_years],
    "Interest Earned (₹)": [f"{x:,.2f}" for x in interest_over_years],
    "Maturity Amount (₹)": [f"{x:,.2f}" for x in maturity_over_years],
})

st.dataframe(table_data)

# --- Step 9: Why invest here? ---
st.subheader("Why Invest in This Fund?")
st.markdown(f"""
- **Risk Level:** {risk_profile}  
- **5-Year CAGR:** {cagr*100:.2f}% (approx.)  
- **Diversified asset allocation** to balance growth and safety.  
- Regularly monitored by professional fund managers.  
- Suitable for investors looking for a **{'stable' if risk_profile=='Low' else 'balanced' if risk_profile=='Moderate' else 'high growth'}** portfolio.  
""")

st.markdown("---")
st.caption("Note: This is a simulation using historical returns. Actual returns may vary. Always consult a financial advisor before investing.")

