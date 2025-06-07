import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------- CONFIG ----------------------- #
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

# ------------------- FUNCTION: CALCULATION ------------------ #
def calculate_growth(amount, rate_percent, years, mode):
    if mode == "Lump Sum":
        return amount * ((1 + rate_percent / 100) ** years)
    else:
        monthly_rate = rate_percent / (12 * 100)
        months = years * 12
        return amount * (((1 + monthly_rate) ** months - 1) * (1 + monthly_rate)) / monthly_rate

# ------------------ FUNCTION: ML MODEL ------------------ #
def train_predict_model(df, target_col):
    df = df.copy()
    features = ['NAV', '1Y_Return', '3Y_Return', '5Y_Return']
    df = df.dropna(subset=features + [target_col])
    df['Risk_Code'] = LabelEncoder().fit_transform(df['Risk'])
    X = df[features + ['Risk_Code']]
    y = df[target_col]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['Predicted_Return'] = model.predict(X)
    return df

# ------------------- MAIN APP ------------------ #
df = load_data()

if df is not None and not df.empty:
    st.title("💼 Mutual Fund Recommender Pro")

    st.markdown("""
    🚀 **Empower your financial journey!** Select your investment preferences and let our smart AI recommend the best mutual funds tailored to your goals.
    """)

    st.sidebar.header("📊 Investment Preferences")
    invest_mode = st.sidebar.radio("Select Investment Type:", ["SIP", "Lump Sum"])
    amount = st.sidebar.number_input("Investment Amount (₹)", min_value=500, step=500, value=5000)
    tenure = st.sidebar.slider("Investment Tenure (Years)", 1, 30, 5)
    risk = st.sidebar.selectbox("Select Risk Appetite", ['Low', 'Moderate', 'High'])

    # Choose target return column
    if tenure <= 1:
        target_col = "1Y_Return"
    elif tenure <= 3:
        target_col = "3Y_Return"
    else:
        target_col = "5Y_Return"

    # ML model predictions
    df_model = train_predict_model(df, target_col)
    df_filtered = df_model[df_model['Risk'].str.lower() == risk.lower()]
    top_funds = df_filtered.sort_values(by='Predicted_Return', ascending=False).head(5).copy()

    if top_funds.empty:
        st.error("No mutual funds match your selected criteria.")
    else:
        # Calculate growth
        projections = []
        for _, row in top_funds.iterrows():
            predicted_rate = row['Predicted_Return']
            projected = calculate_growth(amount, predicted_rate, tenure, invest_mode)
            invested = amount * 12 * tenure if invest_mode == "SIP" else amount
            gain = projected - invested
            projections.append({
                "Fund": row['Scheme Name'],
                "NAV": row['NAV'],
                "Predicted Rate": round(predicted_rate, 2),
                "Invested": invested,
                "Projected Return": round(projected, 2),
                "Gain": round(gain, 2)
            })

        proj_df = pd.DataFrame(projections)
        best_fund = proj_df.sort_values(by="Gain", ascending=False).iloc[0]

        # ---------------- HIGHLIGHT SECTION ---------------- #
        st.markdown("## 🏆 Best Fund Recommendation")
        st.success(f"""
        **✅ Recommended Fund: `{best_fund['Fund']}`**
        
        📈 **Projected Return:** ₹{best_fund['Projected Return']:,.2f}  
        💰 **Invested Amount:** ₹{best_fund['Invested']:,.2f}  
        📊 **Estimated Gain:** ₹{best_fund['Gain']:,.2f}  
        🧠 **Why this fund?**  
        - Highest projected gain using AI-based predictions.  
        - Matches your selected **{risk}** risk appetite.  
        - Strong NAV and consistent returns over {tenure} years.
        """)

        # ---------------- VISUALIZATIONS ---------------- #
        st.markdown("### 📈 Investment Breakdown")
        pie_data = pd.DataFrame({
            "Category": ["Invested", "Gain"],
            "Value": [best_fund['Invested'], best_fund['Gain']]
        })
        st.plotly_chart(px.pie(pie_data, names="Category", values="Value", title=f"{best_fund['Fund']} - Investment Composition"), use_container_width=True)

        # ---------------- ALTERNATIVES ---------------- #
        st.markdown("### 🔄 Other Top Options")
        st.dataframe(proj_df.set_index("Fund").drop(index=best_fund['Fund']), use_container_width=True)

        # ---------------- BAR CHART ---------------- #
        bar_fig = px.bar(proj_df, x="Fund", y="Projected Return", color="Fund", title="Projected Final Value for Top Funds", text_auto=".2s")
        st.plotly_chart(bar_fig, use_container_width=True)

        # ---------------- SUMMARY ---------------- #
        st.markdown("### 📄 Summary Sheet")
        st.dataframe(proj_df.set_index("Fund"), use_container_width=True)

        st.markdown("> ℹ️ *Predictions are AI-estimated and based on historical performance. Actual outcomes may differ.*")
else:
    st.error("⚠️ Failed to load data. Please check your CSV file and try again.")

# ---------------------- FOOTER ---------------------- #
st.markdown("""
---
🔐 **Disclaimer**: This is a demo application for educational purposes. Please consult a certified financial advisor before making any investment decisions.

Made with ❤️ by [YourName or Company]
""")
