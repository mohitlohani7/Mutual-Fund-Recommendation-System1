import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 📂 Load preprocessed mutual fund data
try:
    df = pd.read_csv("data/preprocessed_mutual_funds.csv")
except FileNotFoundError:
    print("❌ File not found. Please check the file path: 'data/preprocessed_mutual_funds.csv'")
    exit()

# ✅ Select relevant features for clustering
features = ['returns_1yr', 'returns_3yr', 'returns_5yr', 'risk_level_encoded']
if not all(col in df.columns for col in features):
    print(f"❌ One or more required columns missing in data: {features}")
    exit()

X = df[features]

# 🔄 Normalize the data for KMeans
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📊 Elbow Method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 📈 Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS (Within Cluster Sum of Squares)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ Apply KMeans clustering with selected k
optimal_k = 4  # Modify this based on the elbow curve
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 📋 Show cluster distribution
print("\n🔹 Cluster counts:")
print(df['cluster'].value_counts())

# 🔍 Recommend similar mutual funds
target_fund = "Axis Bluechip Fund Direct Plan Growth"  # Replace if not in dataset

if 'scheme_name' not in df.columns:
    print("❌ Column 'scheme_name' not found in dataset.")
    exit()

if target_fund in df['scheme_name'].values:
    target_cluster = df[df['scheme_name'] == target_fund]['cluster'].values[0]
    recommendations = df[(df['cluster'] == target_cluster) & (df['scheme_name'] != target_fund)]

    print(f"\n✅ Top 5 Similar Mutual Funds to '{target_fund}':\n")
    print(recommendations[['scheme_name', 'returns_1yr', 'returns_3yr', 'returns_5yr']].head())

    # 💾 Save recommendations
    recommendations.to_csv("data/fund_recommendations.csv", index=False)
    print("\n📁 Recommendations saved to 'data/fund_recommendations.csv'")
else:
    print(f"\n⚠️ Fund '{target_fund}' not found in dataset. No recommendations saved.")
