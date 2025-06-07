import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
        return amount * ((1 + rate_percent / 100) ** years)
    else:
        monthly_rate = rate_percent / (12 * 100)
        months = years * 12
        return amount * (((1 + monthly_rate) ** months - 1) * (1 + monthly_rate)) / monthly_rate

# --------------------- FUNCTION: PREDICTION ------------------ #
def train_predict_model(df, target_col):
    df = df.copy()
    features = ['NAV', '1Y_Return', '3Y_Return', '5Y_Return']
    df = df.dropna(subset=features + [target_col])

    le = LabelEncoder()
    df['Risk_Code'] = le.fit_transform(df['Risk'])

    X = df[features + ['Risk_Code']]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df['Predicted_Return'] = model.predict(X)
    return df

# --------------------- MAIN APP ----------------------------- #
df = load_data()

if df is not None and not df.empty:
    st.title("💼 Mutual Fund Recommender Pro")

    st.markdown("""
    Empower your financial journey by investing smartly. Choose between **SIP** or **Lump Sum**, select your risk appetite,
    and get AI-powered mutual fund recommendations using machine learning to forecast potential returns.
    """)

    st.sidebar.header("📊 Investment Preferences")
    invest_mode = st.sidebar.radio("Select Investment Type:", ["SIP", "Lump Sum"])
    amount = st.sidebar.number_input(f"Monthly Amount (₹)" if invest_mode == "SIP" else "One-time Amount (₹)", min_value=500, step=500, value=5000)
    tenure = st.sidebar.slider("Investment Tenure (Years)", min_value=1, max_value=30, value=5)
    risk = st.sidebar.selectbox("Select Risk Appetite", options=['Low', 'Moderate', 'High'])

    # Map tenure to historical return column
    if tenure <= 1:
        target_col = "1Y_Return"
    elif tenure <= 3:
        target_col = "3Y_Return"
    else:
        target_col = "5Y_Return"

    # Train model and predict returns
    df_model = train_predict_model(df, target_col)
    df_filtered = df_model[df_model['Risk'].str.lower() == risk.lower()]
    top_funds = df_filtered.sort_values(by='Predicted_Return', ascending=False).head(5).copy()

    if top_funds.empty:
        st.error("No mutual funds match your selected risk level.")
    else:
        st.subheader(f"🏆 Top {len(top_funds)} AI-Recommended Mutual Funds for {risk} Risk")
        st.dataframe(top_funds[['Scheme Name', 'NAV', 'Predicted_Return', 'Risk']], use_container_width=True)

        # Investment projection
        projections = []
        for _, row in top_funds.iterrows():
            predicted_rate = row['Predicted_Return']
            projected = calculate_growth(amount, predicted_rate, tenure, invest_mode)
            total_invested = amount * tenure * 12 if invest_mode == "SIP" else amount
            gain = projected - total_invested
            projections.append({
                "Fund": row['Scheme Name'],
                "Invested": total_invested,
                "Return": round(projected, 2),
                "Gain": round(gain, 2),
                "Rate": predicted_rate
            })

        proj_df = pd.DataFrame(projections)

        # Fund selection and pie chart
        selected_fund = st.selectbox("Select a Fund to Visualize Growth", proj_df['Fund'])
        sel_row = proj_df[proj_df['Fund'] == selected_fund].iloc[0]
        pie_data = pd.DataFrame({
            "Label": ["Invested Amount", "Estimated Gain"],
            "Value": [sel_row['Invested'], sel_row['Gain']]
        })

        pie_fig = px.pie(pie_data, names='Label', values='Value', title=f"{selected_fund} - Investment Composition",
                         color_discrete_sequence=['#636EFA', '#00CC96'])
        st.plotly_chart(pie_fig, use_container_width=True)

        # Bar chart of all funds
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

        # Summary sheet
        st.subheader("📄 Investment Summary Sheet")
        st.dataframe(proj_df.set_index("Fund"), use_container_width=True)

        st.markdown("""
        > 💡 *Returns are predicted using machine learning models trained on historical fund performance. Actual results may vary.*
        """)
else:
    st.error("⚠️ Failed to load mutual fund data. Please check the CSV file path and structure.")

# ---------------------- FOOTER ---------------------- #
st.markdown("""
---
**🔐 Disclaimer:** This dashboard is for educational and demo purposes only. Always consult a certified financial advisor before investing.

Built with ❤️ using Streamlit and AI
""")
