import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#                   ****** 1] Pre processing ******
data = pd.read_csv('sales_data_sample.csv', encoding='latin1')  # remove unicode error
print(data.head())

#select numeric colums for clustering
numeric_data1 = data.select_dtypes(include=['int64','float64']).columns
x = data[numeric_data1]

#standardize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#         ***** 2] Elbow Method to determine optional k for K-Means *****
# The Elbow Method helps us decide how many clusters (k) we should make in K-Means.
wcss = []     # This makes an empty list named wcss.
for i in range(1,11):       # So it tests: what happens if we make 1 group, 2 groups, 3 groups... up to 10 groups?
    kmeans = KMeans(n_clusters=i, random_state=42)    # This creates a K-Means model that will make i clusters.  Example: when i=3, it will try to divide the data into 3 clusters.
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss,marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#         ***** 3] Apply K-Means clustering *****
k=4
kmeans = KMeans(n_clusters=k, random_state=42)
data['KMeans_cluster'] = kmeans.fit_predict(x_scaled)
print(data.head())    # Shows first 5 rows of dataset with addinal feature which shows in which cluster the above data point comes

# # Hierarchical Clustering
# linked = linkage(x_scaled, method='ward')
#
# plt.figure(figsize=(10,7))
# dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
# plt.title('Hierarchica; Clustering Dendrogram')
# plt.show()
#
# # Assign clusters (cut dendrogram ak k cluters)
# data['Hier_Cluster'] = fcluster(linked, k, criterion='maxclust')
# print(data.head())