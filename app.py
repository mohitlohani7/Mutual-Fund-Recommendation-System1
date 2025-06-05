import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Mutual Fund Investment Dashboard", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/mutual_funds_enriched.csv', sep=';')
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None and not df.empty:
    st.title("📈 Mutual Fund Investment Dashboard")
    st.markdown(
        """
        Invest smartly by comparing mutual funds based on your risk appetite and investment duration.
        """
    )

    # Sidebar inputs
    st.sidebar.header("Investment Preferences")
    amount = st.sidebar.number_input("Investment Amount (₹)", min_value=1000, step=1000, value=10000)
    duration = st.sidebar.selectbox("Investment Duration (Years)", options=[1, 3, 5], index=1)
    risk_appetite = st.sidebar.multiselect(
        "Select Risk Appetite",
        options=df['Risk'].unique(),
        default=list(df['Risk'].unique())
    )

    # Filter data based on risk appetite
    filtered_df = df[df['Risk'].isin(risk_appetite)]

    if filtered_df.empty:
        st.warning("No mutual funds match the selected risk appetite.")
    else:
        # Select return column based on duration
        return_col = f"{duration}Y_Return"
        if return_col not in filtered_df.columns:
            st.error(f"Return data for {duration} years not available.")
        else:
            # Sort by return descending
            filtered_df = filtered_df.sort_values(by=return_col, ascending=False)

            st.subheader(f"Funds filtered by risk {risk_appetite} and sorted by {duration} year return")

            # Display top 10 funds
            st.dataframe(
                filtered_df[['Scheme Name', 'NAV', return_col, 'Risk']].rename(
                    columns={return_col: f"{duration} Year Return (%)"}
                ),
                use_container_width=True
            )

            # Recommend top 3 funds
            top3 = filtered_df.head(3)

            st.markdown("### 🔥 Top 3 Recommended Funds:")
            for idx, row in top3.iterrows():
                st.markdown(
                    f"**{row['Scheme Name']}** | NAV: ₹{row['NAV']:.2f} | "
                    f"{duration} Year Return: {row[return_col]:.2f}% | Risk: {row['Risk']}"
                )

            # NAV Trend Chart (Dummy static chart as we only have one date currently)
            st.subheader("NAV Trend Chart (Demo)")
            st.markdown(
                """
                *(NAV trend for selected funds over time will be displayed here when historical NAV data is available.)*
                """
            )

            # Plot example NAV trend for top 3 funds (dummy data)
            dates = pd.date_range(start="2021-01-01", periods=12, freq='M')
            fig, ax = plt.subplots(figsize=(10, 5))
            for idx, row in top3.iterrows():
                nav_trend = row['NAV'] * (1 + np.linspace(-0.05, 0.05, len(dates)))  # Dummy fluctuating NAV
                ax.plot(dates, nav_trend, label=row['Scheme Name'])

            ax.set_title("NAV Trend (Last 12 Months, Simulated)")
            ax.set_xlabel("Date")
            ax.set_ylabel("NAV (₹)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Investment projection (simple compound interest)
            st.subheader("Investment Projection")

            def projected_value(principal, rate_percent, years):
                return principal * (1 + rate_percent/100)**years

            for idx, row in top3.iterrows():
                proj_val = projected_value(amount, row[return_col], duration)
                st.markdown(
                    f"**If you invest ₹{amount:,} in {row['Scheme Name']} for {duration} years at "
                    f"{row[return_col]:.2f}% annual return, your projected value: ₹{proj_val:,.2f}**"
                )

else:
    st.error("Failed to load mutual funds data. Please check the CSV file and path.")

# Footer with disclaimers
st.markdown("---")
st.markdown(
    """
    **Disclaimer:** This dashboard provides simulated data and is for educational/demo purposes only.
    Please consult a certified financial advisor before making investment decisions.
    """
)
