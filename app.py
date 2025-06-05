import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(page_title="Mutual Fund Investment Dashboard", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/mutual_funds_enriched.csv', sep=';')
        df.columns = df.columns.str.strip()
        # Ensure numeric columns are properly converted
        numeric_cols = ['NAV', '1Y_Return', '3Y_Return', '5Y_Return']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['Scheme Name', 'NAV'])
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

df = load_data()

def compound_interest(principal, rate, years):
    """Calculate projected investment value with compound interest."""
    return principal * (1 + rate / 100) ** years

if df is not None and not df.empty:
    st.title("📊 Mutual Fund Investment Dashboard")

    st.markdown("""
    Welcome! Use the sidebar to filter mutual funds based on your investment preferences.
    This dashboard helps you compare funds by risk, returns, and projects your investment growth.
    """)

    # Sidebar inputs
    st.sidebar.header("Set Your Investment Preferences")
    investment_amount = st.sidebar.number_input("Investment Amount (₹)", min_value=1000, step=1000, value=10000, format="%d")
    investment_duration = st.sidebar.selectbox("Investment Duration (Years)", options=[1, 3, 5], index=1)
    risk_options = df['Risk'].dropna().unique() if 'Risk' in df.columns else []
    selected_risks = st.sidebar.multiselect("Risk Appetite", options=risk_options, default=list(risk_options))

    # Filter by risk appetite if selected
    filtered_df = df[df['Risk'].isin(selected_risks)] if selected_risks else df.copy()

    # Validate return column availability
    return_col = f"{investment_duration}Y_Return"
    if return_col not in filtered_df.columns:
        st.error(f"Return data for {investment_duration} years is not available.")
        st.stop()

    # Remove rows with NaN in return_col
    filtered_df = filtered_df.dropna(subset=[return_col])

    if filtered_df.empty:
        st.warning("No mutual funds match your selected filters.")
        st.stop()

    # Sort funds by return
    filtered_df = filtered_df.sort_values(by=return_col, ascending=False)

    # Layout with tabs
    tab1, tab2, tab3 = st.tabs(["Fund List", "Top 3 Recommendations", "Investment Projection"])

    with tab1:
        st.subheader("Filtered Mutual Funds")
        st.dataframe(
            filtered_df[['Scheme Name', 'NAV', return_col, 'Risk']].rename(
                columns={
                    'NAV': 'Net Asset Value (₹)',
                    return_col: f"{investment_duration} Year Return (%)",
                    'Risk': 'Risk Category'
                }
            ).reset_index(drop=True),
            use_container_width=True
        )

    with tab2:
        st.subheader("🔥 Top 3 Recommended Funds")

        top3 = filtered_df.head(3).reset_index(drop=True)
        for idx, row in top3.iterrows():
            st.markdown(
                f"**{idx + 1}. {row['Scheme Name']}**\n"
                f"- NAV: ₹{row['NAV']:.2f}\n"
                f"- {investment_duration} Year Return: {row[return_col]:.2f}%\n"
                f"- Risk: {row['Risk']}"
            )

        # Interactive NAV Trend Chart (Simulated for top 3 funds)
        st.subheader("NAV Trend (Simulated Last 12 Months)")

        dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='M')
        fig = go.Figure()
        for _, row in top3.iterrows():
            nav_sim = row['NAV'] * (1 + np.linspace(-0.05, 0.05, len(dates)))  # simulated NAV trend
            fig.add_trace(go.Scatter(x=dates, y=nav_sim, mode='lines+markers', name=row['Scheme Name']))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="NAV (₹)",
            legend_title="Scheme Name",
            template="plotly_white",
            height=450,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Investment Projection")

        for _, row in filtered_df.head(5).iterrows():
            proj_value = compound_interest(investment_amount, row[return_col], investment_duration)
            st.markdown(
                f"**{row['Scheme Name']}**: If you invest ₹{investment_amount:,} for "
                f"{investment_duration} years at an annual return of {row[return_col]:.2f}%, "
                f"your projected value will be ₹{proj_value:,.2f}."
            )

else:
    st.error("Failed to load mutual funds data or the dataset is empty. Please check your CSV file and path.")

# Footer
st.markdown("---")
st.markdown(
    """
    <small>
    **Disclaimer:** This dashboard is for educational and demonstration purposes only.
    Please consult a financial advisor before making any investment decisions.
    </small>
    """,
    unsafe_allow_html=True
)
