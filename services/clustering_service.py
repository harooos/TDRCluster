"""
物理切分服务
"""
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
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
                is_noise=False,
            )
            print(f"HDBSCAN未找到稳定稠密簇，回退为单一micro-cluster: {fallback_cluster.id}")
            return [fallback_cluster], []

        for label, indices in sorted(label_to_indices.items(), key=lambda item: item[0]):
            cluster_queries = [queries[idx] for idx in indices]
            cluster = self._build_cluster(
                task=task,
                queries=cluster_queries,
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
        is_noise: bool,
    ) -> MicroCluster:
        cluster_id = self._generate_cluster_id()
        cluster = MicroCluster(
            id=cluster_id,
            parent_task_id=task.id,
            parent_cluster_id=task.parent_cluster_id,
            depth=task.depth,
            queries=queries,
            is_noise=is_noise,
            judge=None,
        )
        initial_evidence = self._build_evidence_package(
            queries=queries,
            sample_budget=self.config.get("max_samples_per_cluster", 16),
        )
        cluster.add_evidence_round(
            round_index=1,
            mode="initial",
            groups=initial_evidence["groups"],
            query_ids=initial_evidence["query_ids"],
        )
        return cluster

    def _generate_cluster_id(self) -> str:
        self.cluster_counter += 1
        return f"mc-{self.cluster_counter:04d}"

    def sample_additional_evidence(
        self,
        cluster: MicroCluster,
        sample_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._build_evidence_package(
            queries=cluster.queries,
            sample_budget=sample_budget or self.config.get("max_samples_per_cluster", 16),
            excluded_query_ids=set(cluster.sample_query_ids),
        )

    def _build_evidence_package(
        self,
        queries: List[Query],
        sample_budget: int,
        excluded_query_ids: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        excluded_query_ids = excluded_query_ids or set()
        candidate_indices = [
            index for index, query in enumerate(queries)
            if query.id not in excluded_query_ids
        ]
        if not candidate_indices or sample_budget <= 0:
            return {
                "query_ids": [],
                "sample_texts": [],
                "groups": {},
                "coverage_ratio": len(excluded_query_ids) / max(1, len(queries)),
            }

        total_budget = min(sample_budget, len(candidate_indices))
        budget = self._allocate_sample_budget(total_budget)
        embeddings = np.array([query.embedding for query in queries], dtype=float)

        representative = self._select_representative_indices(
            embeddings,
            candidate_indices,
            budget["representative"],
        )
        remaining = self._remaining_indices(candidate_indices, representative)

        diverse = self._select_diverse_indices(
            embeddings,
            remaining,
            budget["diverse"],
            anchor_indices=representative,
        )
        remaining = self._remaining_indices(remaining, diverse)

        boundary = self._select_boundary_indices(
            embeddings,
            remaining,
            budget["boundary"],
        )
        remaining = self._remaining_indices(remaining, boundary)

        selected_indices = representative + diverse + boundary
        additional = []
        if len(selected_indices) < total_budget:
            additional = self._select_fill_indices(
                embeddings,
                remaining,
                total_budget - len(selected_indices),
                anchor_indices=selected_indices or representative,
            )
            selected_indices.extend(additional)

        groups = {
            "representative": [queries[index].content for index in representative],
            "diverse": [queries[index].content for index in diverse],
            "boundary": [queries[index].content for index in boundary],
            "additional": [queries[index].content for index in additional],
        }

        ordered_group_names = ("representative", "diverse", "boundary", "additional")
        sample_texts = []
        for group_name in ordered_group_names:
            sample_texts.extend(groups[group_name])

        selected_query_ids = [queries[index].id for index in selected_indices]

        return {
            "query_ids": selected_query_ids,
            "sample_texts": sample_texts,
            "groups": {name: values for name, values in groups.items() if values},
            "coverage_ratio": (
                len(excluded_query_ids) + len(selected_indices)
            ) / max(1, len(queries)),
            "novelty_score": self._compute_novelty_score(
                queries=queries,
                embeddings=embeddings,
                new_query_ids=selected_query_ids,
                reference_query_ids=list(excluded_query_ids),
            ),
        }

    def _allocate_sample_budget(self, total_budget: int) -> Dict[str, int]:
        if total_budget <= 0:
            return {"representative": 0, "diverse": 0, "boundary": 0}
        if total_budget == 1:
            return {"representative": 1, "diverse": 0, "boundary": 0}
        if total_budget == 2:
            return {"representative": 1, "diverse": 1, "boundary": 0}
        if total_budget == 3:
            return {"representative": 1, "diverse": 1, "boundary": 1}

        representative = max(1, round(total_budget * 0.375))
        diverse = max(1, round(total_budget * 0.375))
        boundary = max(1, total_budget - representative - diverse)

        while representative + diverse + boundary > total_budget:
            if representative >= diverse and representative >= boundary and representative > 1:
                representative -= 1
            elif diverse >= boundary and diverse > 1:
                diverse -= 1
            elif boundary > 1:
                boundary -= 1
            else:
                break

        while representative + diverse + boundary < total_budget:
            if representative <= diverse:
                representative += 1
            else:
                diverse += 1

        return {
            "representative": representative,
            "diverse": diverse,
            "boundary": boundary,
        }

    def _select_representative_indices(
        self,
        embeddings: np.ndarray,
        candidate_indices: List[int],
        count: int,
    ) -> List[int]:
        if count <= 0 or not candidate_indices:
            return []

        centroid = embeddings[candidate_indices].mean(axis=0)
        distances = np.linalg.norm(embeddings[candidate_indices] - centroid, axis=1)
        ranked = np.array(candidate_indices)[np.argsort(distances)]
        return ranked[:count].tolist()

    def _select_diverse_indices(
        self,
        embeddings: np.ndarray,
        candidate_indices: List[int],
        count: int,
        anchor_indices: Optional[List[int]] = None,
    ) -> List[int]:
        if count <= 0 or not candidate_indices:
            return []

        selected = []
        remaining = list(candidate_indices)
        anchor_indices = list(anchor_indices or [])

        if anchor_indices:
            reference_vectors = embeddings[anchor_indices]
        else:
            centroid = embeddings[remaining].mean(axis=0)
            first_index = remaining[int(np.argmax(np.linalg.norm(embeddings[remaining] - centroid, axis=1)))]
            selected.append(first_index)
            remaining.remove(first_index)
            reference_vectors = embeddings[[first_index]]

        while remaining and len(selected) < count:
            distances = [
                (
                    index,
                    float(np.min(np.linalg.norm(reference_vectors - embeddings[index], axis=1))),
                )
                for index in remaining
            ]
            next_index = max(distances, key=lambda item: item[1])[0]
            selected.append(next_index)
            remaining.remove(next_index)
            reference_vectors = np.vstack([reference_vectors, embeddings[next_index]])

        return selected

    def _select_boundary_indices(
        self,
        embeddings: np.ndarray,
        candidate_indices: List[int],
        count: int,
    ) -> List[int]:
        if count <= 0 or not candidate_indices:
            return []

        centroid = embeddings.mean(axis=0)
        centroid_distances = np.linalg.norm(embeddings[candidate_indices] - centroid, axis=1)
        all_local_sparsity = self._compute_local_sparsity(embeddings)
        local_sparsity = all_local_sparsity[candidate_indices]
        combined_score = self._normalize_scores(centroid_distances) + self._normalize_scores(local_sparsity)
        ranked = np.array(candidate_indices)[np.argsort(-combined_score)]
        return ranked[:count].tolist()

    def _select_fill_indices(
        self,
        embeddings: np.ndarray,
        candidate_indices: List[int],
        count: int,
        anchor_indices: Optional[List[int]] = None,
    ) -> List[int]:
        return self._select_diverse_indices(
            embeddings=embeddings,
            candidate_indices=candidate_indices,
            count=count,
            anchor_indices=anchor_indices,
        )

    def _remaining_indices(self, original: List[int], selected: List[int]) -> List[int]:
        selected_set = set(selected)
        return [index for index in original if index not in selected_set]

    def _compute_local_sparsity(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) <= 2:
            return np.ones(len(embeddings))

        neighbor_count = min(max(2, int(np.sqrt(len(embeddings)))), len(embeddings) - 1)
        model = NearestNeighbors(
            n_neighbors=neighbor_count + 1,
            metric="cosine",
        )
        model.fit(embeddings)
        distances, _ = model.kneighbors(embeddings)
        return distances[:, 1:].mean(axis=1)

    def _normalize_scores(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values

        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if np.isclose(min_value, max_value):
            return np.ones(len(values))

        return (values - min_value) / (max_value - min_value)

    def _compute_novelty_score(
        self,
        queries: List[Query],
        embeddings: np.ndarray,
        new_query_ids: List[str],
        reference_query_ids: List[str],
    ) -> float:
        if not new_query_ids:
            return 0.0
        if not reference_query_ids:
            return 1.0

        id_to_index = {query.id: index for index, query in enumerate(queries)}
        new_indices = [id_to_index[query_id] for query_id in new_query_ids if query_id in id_to_index]
        reference_indices = [
            id_to_index[query_id]
            for query_id in reference_query_ids
            if query_id in id_to_index
        ]
        if not new_indices:
            return 0.0
        if not reference_indices:
            return 1.0

        new_vectors = self._l2_normalize(embeddings[new_indices])
        reference_vectors = self._l2_normalize(embeddings[reference_indices])
        cosine_similarity = np.clip(new_vectors @ reference_vectors.T, -1.0, 1.0)
        cosine_distance = 1.0 - cosine_similarity
        return float(np.min(cosine_distance, axis=1).mean())

    def _l2_normalize(self, values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return values / norms

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
