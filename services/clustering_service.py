"""
物理切分服务
"""
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from config.config_loader import CONFIG
from core.state import Query, SplitTask, MicroCluster

try:
    import hdbscan
except ImportError:  # pragma: no cover - 运行时依赖
    hdbscan = None

try:
    import umap
except ImportError:  # pragma: no cover - 运行时依赖
    umap = None


class ClusteringService:
    """基于UMAP + HDBSCAN的物理切分服务"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_counter = 0
        self.config = CONFIG.get("clustering", {})

    def perform_clustering(
        self,
        task: SplitTask,
        total_queries: int,
    ) -> Tuple[List[MicroCluster], List[MicroCluster]]:
        """执行一次物理切分，返回可评审簇和残差簇"""
        queries = task.queries
        if not queries:
            print("警告: 输入queries为空")
            return [], []

        min_cluster_size = self.calculate_min_cluster_size(total_queries)
        print(
            f"开始物理切分: task={task.id}, depth={task.depth}, "
            f"queries={len(queries)}, min_cluster_size={min_cluster_size}"
        )

        if len(queries) <= max(3, min_cluster_size):
            dense_cluster = self._build_cluster(
                task=task,
                queries=queries,
                embeddings=np.array([q.embedding for q in queries]),
                is_noise=False,
            )
            print(f"查询量较小，直接作为单一micro-cluster输出: {dense_cluster.id}")
            return [dense_cluster], []

        self._ensure_backends()

        embeddings = np.array([q.embedding for q in queries], dtype=float)
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        reduced_embeddings = self._reduce_embeddings(scaled_embeddings)
        labels = self._run_hdbscan(reduced_embeddings, min_cluster_size)

        label_to_indices = {}
        for index, label in enumerate(labels):
            label_to_indices.setdefault(int(label), []).append(index)

        review_clusters: List[MicroCluster] = []
        residual_clusters: List[MicroCluster] = []

        dense_labels = [label for label in label_to_indices if label != -1]
        if not dense_labels:
            fallback_cluster = self._build_cluster(
                task=task,
                queries=queries,
                embeddings=reduced_embeddings,
                is_noise=False,
            )
            print(f"HDBSCAN未找到稳定稠密簇，回退为单一micro-cluster: {fallback_cluster.id}")
            return [fallback_cluster], []

        for label, indices in sorted(label_to_indices.items(), key=lambda item: item[0]):
            cluster_queries = [queries[idx] for idx in indices]
            cluster_embeddings = reduced_embeddings[indices]
            cluster = self._build_cluster(
                task=task,
                queries=cluster_queries,
                embeddings=cluster_embeddings,
                is_noise=(label == -1),
            )

            if label == -1:
                residual_clusters.append(cluster)
            else:
                review_clusters.append(cluster)

        self._print_cluster_summary(review_clusters, residual_clusters)
        return review_clusters, residual_clusters

    def _ensure_backends(self):
        if umap is None or hdbscan is None:
            raise RuntimeError(
                "UMAP/HDBSCAN 依赖未安装，请先执行 `pip install -r requirements.txt`"
            )

    def _reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        n_samples = len(embeddings)
        if n_samples <= 3:
            return embeddings

        umap_config = self.config.get("umap", {})
        n_neighbors = min(umap_config.get("n_neighbors", 15), n_samples - 1)
        n_components = min(umap_config.get("n_components", 10), max(2, n_samples - 1))

        reducer = umap.UMAP(
            n_neighbors=max(2, n_neighbors),
            n_components=max(2, n_components),
            min_dist=umap_config.get("min_dist", 0.0),
            metric=umap_config.get("metric", "cosine"),
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    def _run_hdbscan(self, embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
        hdbscan_config = self.config.get("hdbscan", {})
        configured_min_samples = hdbscan_config.get("min_samples")
        if configured_min_samples is None:
            configured_min_samples = max(2, min_cluster_size // 2)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            min_samples=max(1, configured_min_samples),
            metric=hdbscan_config.get("metric", "euclidean"),
            cluster_selection_method=hdbscan_config.get("cluster_selection_method", "eom"),
        )
        return clusterer.fit_predict(embeddings)

    def _build_cluster(
        self,
        task: SplitTask,
        queries: List[Query],
        embeddings: np.ndarray,
        is_noise: bool,
    ) -> MicroCluster:
        cluster_id = self._generate_cluster_id()
        sample_texts = self._extract_representative_samples(queries, embeddings)
        return MicroCluster(
            id=cluster_id,
            parent_task_id=task.id,
            parent_cluster_id=task.parent_cluster_id,
            depth=task.depth,
            queries=queries,
            sample_texts=sample_texts,
            is_noise=is_noise,
            judge=None,
        )

    def _generate_cluster_id(self) -> str:
        self.cluster_counter += 1
        return f"mc-{self.cluster_counter:04d}"

    def _extract_representative_samples(
        self,
        queries: List[Query],
        embeddings: np.ndarray,
    ) -> List[str]:
        max_samples = self.config.get("max_samples_per_cluster", 8)
        if len(queries) <= max_samples:
            return [query.content for query in queries]

        centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        representative_indices = np.argsort(distances)[:max_samples]
        return [queries[index].content for index in representative_indices]

    def _print_cluster_summary(
        self,
        review_clusters: List[MicroCluster],
        residual_clusters: List[MicroCluster],
    ):
        print(f"物理切分完成: review_clusters={len(review_clusters)}, residual_clusters={len(residual_clusters)}")
        for cluster in review_clusters[:10]:
            print(f"  {cluster.id}: {len(cluster.queries)} 个查询, depth={cluster.depth}")
        for cluster in residual_clusters[:5]:
            print(f"  {cluster.id}: {len(cluster.queries)} 个残差查询")

    def calculate_min_cluster_size(self, total_queries: int) -> int:
        """根据全局配置计算最小簇大小"""
        min_config = self.config.get("min_cluster_size", {"absolute": 10, "ratio": 0.005})
        ratio_based = int(total_queries * min_config.get("ratio", 0.005))
        return max(min_config.get("absolute", 10), ratio_based)
