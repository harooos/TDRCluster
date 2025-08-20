"""
聚类服务
"""
from typing import List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random

from core.state import Query, Cluster


class ClusteringService:
    """聚类服务"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_counter = 0
    
    def perform_clustering(self, queries: List[Query], k_value: int) -> List[Cluster]:
        """执行K-Means聚类"""
        if not queries:
            print("警告: 输入queries为空")
            return []
        
        if len(queries) < k_value:
            print(f"警告: queries数量({len(queries)}) < k_value({k_value})，调整k_value为{len(queries)}")
            k_value = len(queries)
        
        print(f"开始聚类: {len(queries)} 个查询 -> {k_value} 个clusters")
        
        # 提取embedding向量
        embeddings = np.array([q.embedding for q in queries])
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        
        # 执行K-Means聚类
        kmeans = KMeans(
            n_clusters=k_value, 
            random_state=42,     # 硬编码随机种子
            n_init=10,          # 硬编码合理的初始化次数
            max_iter=300        # 硬编码合理的最大迭代次数
        )
        
        try:
            kmeans.fit(scaled_embeddings)
            labels = kmeans.labels_
        except Exception as e:
            print(f"聚类失败: {str(e)}")
            labels = [i % k_value for i in range(len(queries))]
        
        # 按标签分组查询
        clustered_queries = {i: [] for i in range(k_value)}
        for i, query in enumerate(queries):
            clustered_queries[labels[i]].append(query)
        
        # 创建Cluster对象
        clusters = []
        for i in range(k_value):
            cluster_queries = clustered_queries[i]
            if cluster_queries:
                cluster = Cluster(
                    id=self._generate_cluster_id(),
                    queries=cluster_queries,
                    samples=self._extract_samples(cluster_queries),
                    judge=None
                )
                clusters.append(cluster)
        
        print(f"聚类完成: 生成 {len(clusters)} 个有效clusters")
        self._print_cluster_summary(clusters)
        
        return clusters
    
    def _generate_cluster_id(self) -> str:
        """生成唯一的cluster ID"""
        self.cluster_counter += 1
        return f"cluster-{self.cluster_counter}"
    
    def _extract_samples(self, queries: List[Query], max_samples: int = None) -> List[str]:
        """从cluster中提取代表性样本"""
        if max_samples is None:
            from config.config_loader import CONFIG
            max_samples = CONFIG.get('clustering', {}).get('max_samples_per_cluster', 10)
            
        if len(queries) <= max_samples:
            return [q.content for q in queries]
        
        sampled_queries = random.sample(queries, max_samples)
        return [q.content for q in sampled_queries]
    
    def _print_cluster_summary(self, clusters: List[Cluster]):
        """打印聚类结果摘要"""
        for cluster in clusters:
            print(f"  {cluster.id}: {len(cluster.queries)} 个查询")
            samples_preview = cluster.samples[:2]
            for sample in samples_preview:
                sample_short = sample[:40] + '...' if len(sample) > 40 else sample
                print(f"    - {sample_short}")
    
    def calculate_min_cluster_size(self, total_queries: int) -> int:
        """根据全局配置计算最小簇大小"""
        from config.config_loader import CONFIG
        
        min_config = CONFIG.get('clustering', {}).get('min_cluster_size', {'absolute': 10, 'ratio': 0.005})
        ratio_based = int(total_queries * min_config['ratio'])
        return max(min_config['absolute'], ratio_based)
    
