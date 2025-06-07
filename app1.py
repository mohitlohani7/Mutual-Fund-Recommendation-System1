import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------- PAGE CONFIG ----------------------- #
st.set_page_config(page_title="💼 Mutual Fund Recommender Pro", layout="wide")

# ----------------------- LOAD DATA ------------------------ #
@st.cache_data

def load_data():
    df = pd.read_csv("data/mutual_funds_enriched.csv", sep=";")
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "Net Asset Value (NAV)": "NAV",
        "1Y_Return": "1Y_Return",
        "3Y_Return": "3Y_Return",
        "5Y_Return": "5Y_Return"
    }, inplace=True)
    return df.dropna(subset=['NAV', '1Y_Return', '3Y_Return', '5Y_Return', 'Risk'])

# --------------------- FUNCTION: CALCULATION ------------------ #
def calculate_growth(amount, rate_percent, years, mode):
    if mode == "Lump Sum":
        final = amount * ((1 + rate_percent/100) ** years)
        return final
    else:  # SIP
        monthly_rate = rate_percent / (12 * 100)
        months = years * 12
        final = amount * (((1 + monthly_rate) ** months - 1) * (1 + monthly_rate)) / monthly_rate
        return final

# --------------------- MAIN APP ----------------------------- #
df = load_data()

if df is not None and not df.empty:
    st.title("💼 Mutual Fund Recommender Pro")
    st.markdown("""
    Empower your financial journey by investing smartly. Choose between **SIP** or **Lump Sum**, select your risk appetite,
    and get the top mutual fund recommendations with return forecasts and visuals to help guide your decision.
    """)

    st.sidebar.header("📊 Investment Preferences")
    invest_mode = st.sidebar.radio("Select Investment Type:", ["SIP", "Lump Sum"])
    amount = st.sidebar.number_input(f"Monthly Amount (₹)" if invest_mode=="SIP" else "One-time Amount (₹)", min_value=500, step=500, value=5000)
    tenure = st.sidebar.slider("Investment Tenure (Years)", min_value=1, max_value=30, value=5)

    risk = st.sidebar.selectbox("Select Risk Appetite", options=['Low', 'Moderate', 'High'])

    # Map tenure to return column
    if tenure <= 1:
        return_col = "1Y_Return"
    elif tenure <= 3:
        return_col = "3Y_Return"
    else:
        return_col = "5Y_Return"

    # Filter and sort mutual funds
    filtered = df[df['Risk'].str.lower() == risk.lower()].sort_values(by=return_col, ascending=False)
    top_funds = filtered.head(5).copy()

    if top_funds.empty:
        st.error("No mutual funds match your selected risk level.")
    else:
        st.subheader(f"🏆 Top {len(top_funds)} Mutual Funds for {risk} Risk Appetite")
        st.dataframe(top_funds[['Scheme Name', 'NAV', return_col, 'Risk']], use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Investment Growth Overview")

        # Compute projected growth
        projections = []
        for _, row in top_funds.iterrows():
            projected = calculate_growth(amount, row[return_col], tenure, invest_mode)
            total_invested = amount * tenure * 12 if invest_mode == "SIP" else amount
            gain = projected - total_invested
            projections.append({
                "Fund": row['Scheme Name'],
                "Invested": total_invested,
                "Return": round(projected, 2),
                "Gain": round(gain, 2),
                "Rate": row[return_col]
            })

        proj_df = pd.DataFrame(projections)

        # Show Pie Chart of Gain vs Invested
        selected_fund = st.selectbox("Select a Fund to Visualize Growth", proj_df['Fund'])
        sel_row = proj_df[proj_df['Fund'] == selected_fund].iloc[0]
        pie_data = pd.DataFrame({
            "Label": ["Invested Amount", "Estimated Gain"],
            "Value": [sel_row['Invested'], sel_row['Gain']]
        })

        pie_fig = px.pie(pie_data, names='Label', values='Value', title=f"{selected_fund} - Investment Composition",
                         color_discrete_sequence=['#636EFA', '#00CC96'])
        st.plotly_chart(pie_fig, use_container_width=True)

        # Show Bar chart for all top 5 funds
        bar_fig = px.bar(
            proj_df,
            x="Fund",
            y="Return",
            color="Fund",
            title="📊 Projected Final Value per Fund",
            labels={"Return": "Final Value (₹)"},
            text_auto=".2s"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📄 Investment Summary Sheet")
        st.dataframe(proj_df.set_index("Fund"), use_container_width=True)

        st.markdown("""
        > 💡 *Returns shown are based on historical data and estimations. Actual results may vary.*
        """)
else:
    st.error("⚠️ Failed to load mutual fund data. Please check the CSV file path and structure.")

# ---------------------- FOOTER ---------------------- #
st.markdown("""
---
**🔐 Disclaimer:** This dashboard is for educational and demo purposes only. Always consult a certified financial advisor before investing.

Built with ❤️ using Streamlit
""")
