import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Page config
st.set_page_config(page_title="Mutual Fund Investment Dashboard", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/mutual_funds_enriched.csv', sep=';')
        df.columns = df.columns.str.strip()  # Clean column names
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Load data
df = load_data()

if df.empty:
    st.error("Failed to load mutual funds data or the dataset is empty. Please check your CSV file.")
    st.stop()

# Title & description
st.title("📈 Mutual Fund Investment Dashboard")
st.markdown(
    """
    Invest smartly by comparing mutual funds based on your risk appetite and investment duration.
    Use the controls on the left to customize your investment preferences.
    """
)

# Sidebar Inputs
st.sidebar.header("Investment Preferences")
amount = st.sidebar.number_input("Investment Amount (₹)", min_value=1000, step=1000, value=10000, format="%d")
duration = st.sidebar.selectbox("Investment Duration (Years)", options=[1, 3, 5], index=1)

risk_options = df['Risk'].dropna().unique()
risk_appetite = st.sidebar.multiselect("Select Risk Appetite", options=risk_options, default=risk_options)

# Filter data by selected risk appetite
filtered_df = df[df['Risk'].isin(risk_appetite)]

if filtered_df.empty:
    st.warning("No mutual funds match the selected risk appetite.")
    st.stop()

return_col = f"{duration}Y_Return"

if return_col not in filtered_df.columns:
    st.error(f"Return data for {duration} years is not available.")
    st.stop()

filtered_df = filtered_df.dropna(subset=[return_col])
filtered_df = filtered_df.sort_values(by=return_col, ascending=False)

# Display filtered funds
st.subheader(f"Funds filtered by risk {risk_appetite} and sorted by {duration}-year return")

st.dataframe(
    filtered_df[['Scheme Name', 'NAV', return_col, 'Risk']].rename(
        columns={return_col: f"{duration} Year Return (%)", 'NAV': 'Net Asset Value (₹)', 'Risk': 'Risk Category'}
    ),
    use_container_width=True,
)

# Show Top 3 Recommendations
top3 = filtered_df.head(3)

st.markdown("### 🔥 Top 3 Recommended Funds:")
for i, row in top3.iterrows():
    st.markdown(
        f"**{row['Scheme Name']}** | NAV: ₹{row['NAV']:.2f} | "
        f"{duration} Year Return: {row[return_col]:.2f}% | Risk: {row['Risk']}"
    )

# NAV Trend Chart (Simulated)
st.subheader("NAV Trend Chart (Simulated)")

dates = pd.date_range(start="2021-01-01", periods=12, freq='M')
fig, ax = plt.subplots(figsize=(10, 5))

for _, row in top3.iterrows():
    nav_trend = row['NAV'] * (1 + np.linspace(-0.05, 0.05, len(dates)))  # Simulated NAV fluctuations
    ax.plot(dates, nav_trend, label=row['Scheme Name'])

ax.set_title("NAV Trend (Last 12 Months, Simulated)")
ax.set_xlabel("Date")
ax.set_ylabel("NAV (₹)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Investment Projection
st.subheader("Investment Projection")

def projected_value(principal, rate_percent, years):
    return principal * (1 + rate_percent / 100) ** years

for _, row in top3.iterrows():
    proj_val = projected_value(amount, row[return_col], duration)
    st.markdown(
        f"**If you invest ₹{amount:,} in {row['Scheme Name']} for {duration} years at "
        f"{row[return_col]:.2f}% annual return, your projected value will be: ₹{proj_val:,.2f}**"
    )

# Footer disclaimer
st.markdown("---")
st.markdown(
    """
    <small>
    **Disclaimer:** This dashboard provides simulated data for demonstration purposes only.
    Please consult a certified financial advisor before making investment decisions.
    </small>
    """,
    unsafe_allow_html=True,
)
