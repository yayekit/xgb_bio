from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def generate_sample_data(n_samples=100, n_features=2):
    """Generate sample data for clustering."""
    return np.random.rand(n_samples, n_features)

def perform_kmeans_clustering(data, n_clusters=3, random_state=42):
    """Perform K-means clustering on the given data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def visualize_clustering(data, labels, centroids):
    """Visualize the clustering results."""
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.show()

def main():
    # Generate sample data
    X = generate_sample_data()

    # Perform K-means clustering
    labels, centroids = perform_kmeans_clustering(X)

    # Visualize the results
    visualize_clustering(X, labels, centroids)

if __name__ == "__main__":
    main()