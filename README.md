# Mutual-Fund-Recommendation-System1
Sure! Here’s the **updated professional README content** including your deployed app link. You can directly add this to your GitHub README file:

---

# Mutual Fund Recommendation System

## Project Overview

This project is a **Mutual Fund Recommendation System** built using **Streamlit** and **Pandas** in Python. The system allows users to select their investment preferences, such as **risk appetite** and investment type (**SIP or Lump Sum**), and recommends suitable mutual funds from a curated dataset. It also provides **projected returns**, a **maturity calculator**, and interactive **visualizations** to help investors understand their investment growth over time.

---

## Key Features

* **User Input Options:**

  * Choose investment type: **SIP (Systematic Investment Plan)** or **Lump Sum**
  * Input investment amount, duration, and risk level (Low, Moderate, High)

* **Mutual Fund Filtering & Recommendation:**

  * Filters mutual funds based on user's risk appetite
  * Automatically recommends the top mutual fund based on 1-Year Return among filtered funds
  * Provides a clear explanation of why the recommended fund is suitable for the user’s risk profile

* **Return Calculation & Maturity Projection:**

  * Calculates future value of investment using **compound interest formula**
  * Shows **year-wise interest rate**, **principal invested**, and **total maturity amount** in tabular form

* **Visualizations:**

  * Interactive line chart comparing **principal amount** vs **investment value over time**
  * Pie chart showing **asset allocation** for recommended fund (if data available)

* **Clean UI/UX:**

  * Easy-to-use Streamlit interface with conditional inputs based on investment type
  * Clear, concise result presentation for better investor understanding

---

## Live Demo / Deployment

You can try the live application here:

🔗 [Mutual Fund Recommendation System - Deployed App](https://mutual-fund-recommendation-system1-eutccexphammhwdq9falme.streamlit.app/)

---

## How It Works

1. **User Input:**

   * Select risk appetite (Low, Moderate, High)
   * Select investment type (SIP or Lump Sum)
   * Enter investment amount and duration (years)

2. **Fund Filtering:**

   * The system filters mutual funds from the dataset based on the selected risk appetite

3. **Fund Recommendation:**

   * Among filtered funds, the system selects the one with the highest 1-Year Return
   * Displays fund details, including NAV, past returns, and risk level

4. **Investment Calculation:**

   * Using compound interest formula:

     * For Lump Sum: $A = P \times (1 + r)^t$
     * For SIP: $A = P \times \frac{(1 + r)^t - 1}{r} \times (1 + r)$
   * Calculates maturity value and yearly returns

5. **Results Visualization:**

   * Line graph showing growth of investment principal vs total value over time
   * Table showing yearly breakdown of investment, interest, and accumulated amount
   * Explanation on why this fund is recommended based on risk and returns

---

## Project Architecture and Flowchart

```plaintext
+------------------+
|    User Input    | <-- Select Risk Appetite, Investment Type, Amount, Duration
+--------+---------+
         |
         v
+------------------+
| Filter Funds     | <-- Filter dataset by Risk Level matching user input
+--------+---------+
         |
         v
+------------------+
| Recommend Fund   | <-- Select fund with highest 1-Year Return in filtered list
+--------+---------+
         |
         v
+------------------+
| Calculate Returns| <-- Compute future value using compound interest formulas
+--------+---------+
         |
         v
+------------------+
| Visualization    | <-- Display investment growth charts and tables
+--------+---------+
         |
         v
+------------------+
|   User Result    | <-- Show recommended fund and explanation
+------------------+
```

---

### Flowchart Diagram

```
┌─────────────┐
│  User Input │
│ (Risk, Type,│
│  Amount,    │
│  Duration)  │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Filter Fund │
│ by Risk     │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│Recommend    │
│Fund w/Best │
│Return      │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│Calculate    │
│Returns     │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│Visualize    │
│Results     │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│Show Result  │
│and Advice   │
└─────────────┘
```

---

## Project File Structure

```
mutual-fund-recommendation/
│
├── app.py                # Main Streamlit app file
├── mutual_funds.csv      # Dataset CSV file containing mutual fund data
├── README.md             # Project explanation (this file)
├── requirements.txt      # Python dependencies (streamlit, pandas, matplotlib, etc.)
└── assets/
    └── flowchart.png     # Project flowchart image for documentation
```

---

## How to Run

1. Clone repo:

   ```bash
   git clone https://github.com/yourusername/mutual-fund-recommendation.git
   cd mutual-fund-recommendation
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run app:

   ```bash
   streamlit run app.py
   ```

---

## Future Improvements

* Add machine learning model to recommend funds based on historical performance & market conditions
* Include more detailed fund data such as expense ratios, asset allocation, and sector exposure
* Allow user portfolio tracking with multiple funds
* Add authentication & user profiles for personalized recommendations
* Integrate live NAV updates via APIs for real-time data

---

If you want, I can also help prepare the **flowchart image file** and finalize your **app.py** code for your repo. Just ask!
