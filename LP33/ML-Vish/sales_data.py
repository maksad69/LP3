# 🧩 K-Means Clustering on Sales Data (Elbow Method + Segmentation)
import pandas as pd, numpy as np, matplotlib.pyplot as plt, datetime as dt, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

# 1️⃣ Load Dataset
df = pd.read_csv('sales_data_sample.csv', encoding='unicode_escape')
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
df.drop(['ADDRESSLINE1','ADDRESSLINE2','STATE','POSTALCODE','PHONE'], axis=1, inplace=True, errors='ignore')

# 2️⃣ RFM Feature Calculation
snapshot = df['ORDERDATE'].max() + dt.timedelta(days=1)
RFM = df.groupby('CUSTOMERNAME').agg({
    'ORDERDATE': lambda x: (snapshot - x.max()).days,
    'ORDERNUMBER': 'count',
    'SALES': 'sum'
}).rename(columns={'ORDERDATE':'Recency','ORDERNUMBER':'Frequency','SALES':'Monetary'})

# 3️⃣ Normalize Data
X = np.log1p(RFM)
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# 4️⃣ Elbow Method
sse = [KMeans(n_clusters=k, random_state=1).fit(X).inertia_ for k in range(1,11)]
plt.plot(range(1,11), sse, 'bo-')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Clusters (k)'); plt.ylabel('SSE'); plt.show()

# 5️⃣ Apply K-Means (Assume k=5 from elbow)
kmeans = KMeans(n_clusters=5, random_state=1)
RFM['Cluster'] = kmeans.fit_predict(X)

# 6️⃣ Cluster Summary
summary = RFM.groupby('Cluster').mean().round(2)
print("\n📊 Cluster Centers:\n", summary)
print("\nCluster Distribution:\n", RFM['Cluster'].value_counts())

# 7️⃣ Optional: 3D Visualization
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
for c in RFM['Cluster'].unique():
    d = RFM[RFM['Cluster']==c]
    ax.scatter(d['Recency'], d['Frequency'], d['Monetary'], label=f'Cluster {c}', alpha=0.6)
ax.set_xlabel('Recency'); ax.set_ylabel('Frequency'); ax.set_zlabel('Monetary')
ax.set_title('Customer Clusters'); ax.legend(); plt.show()
