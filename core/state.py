from typing import List, Dict, Deque, Any, Optional, TypedDict
from collections import deque
from pydantic import BaseModel, Field

class Query(BaseModel):
    """单个查询对象，包含内容和向量表示"""
    id: str
    content: str
    embedding: List[float]
    
    def __str__(self):
        return f"Query(id={self.id}, content={self.content[:50]}...)"

class Task(BaseModel):
    """聚类任务，包含待处理的查询集合和k值"""
    queries: List[Query]
    k_value: int
    
    def __str__(self):
        return f"Task(queries_count={len(self.queries)}, k_value={self.k_value})"

class Cluster(BaseModel):
    """临时聚类结果，包含查询集合、样本和LLM决策"""
    id: str
    queries: List[Query]
    samples: List[str] = Field(default_factory=list)
    judge: Optional[Dict[str, Any]] = None
    
    def __str__(self):
        return f"Cluster(id={self.id}, queries_count={len(self.queries)})"

class Category(BaseModel):
    """最终的业务类别，包含查询集合和Rich Description"""
    id: str
    description: str  # Rich Description with examples
    queries: List[Query]
    samples: List[str] = Field(default_factory=list)  # Can be empty if using Rich Description
    
    def __str__(self):
        return f"Category(id={self.id}, queries_count={len(self.queries)})"
    
    @property
    def query_count(self) -> int:
        """返回包含的查询数量，用于展示给LLM"""
        return len(self.queries)

# LangGraph的全局状态定义
class GraphState(TypedDict):
    """LangGraph流程的全局状态容器"""
    tasks: Deque[Task]  # 待处理的任务队列
    categories: Dict[str, Category]  # 已确定的最终类别
    clusters_list: Optional[List[Cluster]]  # 当前批次的临时聚类结果
    dataset_name: Optional[str]  # 数据集名称，用于获取特定配置
    # 新增：缓存的全局统计信息
    total_queries: int  # 总查询数（初始化时计算，避免重复计算）
    min_cluster_size: int  # 最小簇大小（基于总查询数计算）