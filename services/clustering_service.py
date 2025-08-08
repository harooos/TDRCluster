from typing import List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

from core.state import Query, Cluster

class ClusteringService:
    def __init__(self):
        pass

    def perform_clustering(self, queries: List[Query], k_value: int) -> List[Cluster]:
        if not queries:
            return []

        # Extract embeddings
        embeddings = np.array([q.embedding for q in queries])

        # Scale embeddings (optional, but good practice for KMeans)
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        kmeans.fit(scaled_embeddings)
        labels = kmeans.labels_

        # Group queries by cluster label
        clustered_queries = {i: [] for i in range(k_value)}
        for i, query in enumerate(queries):
            clustered_queries[labels[i]].append(query)

        # Create Cluster objects
        clusters = []
        for i in range(k_value):
            cluster_queries = clustered_queries[i]
            if cluster_queries:
                # Take a few samples from the cluster for review
                samples = [q.content for q in cluster_queries[:min(5, len(cluster_queries))]]
                cluster = Cluster(
                    id=f"temp-{i}",
                    queries=cluster_queries,
                    samples=samples
                )
                clusters.append(cluster)
        return clusters
