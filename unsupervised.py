from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
X = np.random.rand(100, 2)  # 100 samples with 2 features

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster assignments and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-means Clustering')
plt.show()