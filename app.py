import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(page_title="💼 Mutual Fund Recommender Pro", layout="wide")

# -------------------- LOAD DATA ---------------------- #
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

# ----------------- CALCULATE GROWTH ------------------ #
def calculate_growth(amount, rate_percent, years, mode):
    if mode == "Lump Sum":
        return amount * ((1 + rate_percent / 100) ** years)
    else:
        monthly_rate = rate_percent / (12 * 100)
        months = years * 12
        return amount * (((1 + monthly_rate) ** months - 1) * (1 + monthly_rate)) / monthly_rate

# ----------------- TRAIN PREDICTION MODEL ------------- #
def train_predict_model(df, target_col):
    df = df.copy()
    features = ['NAV', '1Y_Return', '3Y_Return', '5Y_Return']
    df['Risk_Code'] = LabelEncoder().fit_transform(df['Risk'])
    X = df[features + ['Risk_Code']]
    y = df[target_col]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['Predicted_Return'] = model.predict(X)
    return df

# ---------------------- MAIN APP ---------------------- #
df = load_data()

if df is not None and not df.empty:
    st.title("💼 Mutual Fund Recommender Pro")
    st.markdown("""
    > 📊 *Personalized mutual fund recommendations based on your preferences, **powered by Mohit Lohani**.*
    """)

    # ----------- SIDEBAR INPUTS ----------- #
    st.sidebar.header("💰 Investment Preferences")
    invest_mode = st.sidebar.radio("Investment Type", ["SIP", "Lump Sum"])
    amount = st.sidebar.number_input("Investment Amount (₹)", min_value=500, value=5000, step=500)
    tenure = st.sidebar.slider("Tenure (Years)", 1, 30, 5)
    risk = st.sidebar.selectbox("Risk Appetite", ["Low", "Moderate", "High"])

    # ----------- TARGET COLUMN ----------- #
    if tenure <= 1:
        target_col = "1Y_Return"
    elif tenure <= 3:
        target_col = "3Y_Return"
    else:
        target_col = "5Y_Return"

    # ----------- ML MODEL PREDICTION ----------- #
    df_model = train_predict_model(df, target_col)
    df_filtered = df_model[df_model['Risk'].str.lower() == risk.lower()]
    top_funds = df_filtered.sort_values(by="Predicted_Return", ascending=False).head(5).copy()

    if top_funds.empty:
        st.error("⚠️ No mutual funds match your criteria.")
    else:
        # ----------- CALCULATE PROJECTIONS ----------- #
        projections = []
        for _, row in top_funds.iterrows():
            rate = row["Predicted_Return"]
            future_value = calculate_growth(amount, rate, tenure, invest_mode)
            invested = amount * tenure * 12 if invest_mode == "SIP" else amount
            gain = future_value - invested
            projections.append({
                "Fund": row["Scheme Name"],
                "NAV": row["NAV"],
                "Predicted Rate": round(rate, 2),
                "Invested": invested,
                "Projected Return": round(future_value, 2),
                "Gain": round(gain, 2)
            })

        proj_df = pd.DataFrame(projections)
        best_fund = proj_df.sort_values(by="Gain", ascending=False).iloc[0]

        # ----------- BEST FUND HIGHLIGHT ----------- #
        st.markdown("## 🏆 Best Mutual Fund Recommendation")
        st.success(f"""
        **Top Pick:** `{best_fund['Fund']}`  
        💰 Invested: ₹{best_fund['Invested']:,.2f}  
        📈 Projected Return: ₹{best_fund['Projected Return']:,.2f}  
        💹 Estimated Gain: ₹{best_fund['Gain']:,.2f}  
        **Why this?**
        - Highest AI-estimated growth
        - Strong NAV & performance
        - Matches your **{risk}** risk appetite
        """)

        # ----------- USER SELECTION ----------- #
        st.markdown("### 🔄 Explore Other Options")
        fund_options = proj_df["Fund"].tolist()
        selected_fund = st.selectbox("Choose a fund to view details", fund_options, index=fund_options.index(best_fund["Fund"]))
        selected_row = proj_df[proj_df["Fund"] == selected_fund].iloc[0]

        # PIE CHART
        pie_df = pd.DataFrame({
            "Category": ["Invested", "Estimated Gain"],
            "Value": [selected_row["Invested"], selected_row["Gain"]]
        })
        pie_fig = px.pie(pie_df, names="Category", values="Value",
                         title=f"{selected_fund} - Investment Breakdown",
                         color_discrete_sequence=["#636EFA", "#00CC96"])
        st.plotly_chart(pie_fig, use_container_width=True)

        # BAR CHART - All funds
        st.markdown("### 📊 Projected Returns Across Funds")
        bar_fig = px.bar(proj_df, x="Fund", y="Projected Return", color="Fund", text_auto=".2s",
                         labels={"Projected Return": "₹ Projected Final Value"})
        st.plotly_chart(bar_fig, use_container_width=True)

        # SUMMARY TABLE
        st.markdown("### 📋 Investment Summary")
        st.dataframe(proj_df.set_index("Fund"), use_container_width=True)

        st.markdown("> 📌 *Projections are based on historical and predicted trends. Market conditions may vary.*")

else:
    st.error("❌ Failed to load mutual fund data. Please check your CSV and path.")

# ---------------------- FOOTER ---------------------- #
st.markdown("""
---
🔐 **Disclaimer**: This is a demo tool for educational purposes only. Always consult a certified financial advisor before investing.

Made with ❤️ by **Mohit Lohani** using Streamlit
""")
