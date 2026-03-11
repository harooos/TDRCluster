from typing import List, Dict, Deque, Any, Optional, TypedDict
import uuid

from pydantic import BaseModel, Field


def _task_id() -> str:
    return f"task-{uuid.uuid4().hex[:8]}"


class Query(BaseModel):
    """单个查询对象，包含内容和向量表示"""
    id: str
    content: str
    embedding: List[float]

    def __str__(self):
        return f"Query(id={self.id}, content={self.content[:50]}...)"


class SplitTask(BaseModel):
    """BFS队列中的切分任务"""
    id: str = Field(default_factory=_task_id)
    queries: List[Query]
    depth: int = 0
    parent_cluster_id: Optional[str] = None
    split_reason: str = "root"

    def __str__(self):
        return (
            f"SplitTask(id={self.id}, queries_count={len(self.queries)}, "
            f"depth={self.depth}, reason={self.split_reason})"
        )


class MicroCluster(BaseModel):
    """物理切分后的微聚类结果"""
    id: str
    parent_task_id: str
    parent_cluster_id: Optional[str] = None
    depth: int = 0
    queries: List[Query]
    sample_texts: List[str] = Field(default_factory=list)
    is_noise: bool = False
    judge: Optional[Dict[str, Any]] = None

    def __str__(self):
        return (
            f"MicroCluster(id={self.id}, queries_count={len(self.queries)}, "
            f"depth={self.depth}, is_noise={self.is_noise})"
        )

    @property
    def size(self) -> int:
        return len(self.queries)

    @property
    def samples(self) -> List[str]:
        return self.sample_texts


class Category(BaseModel):
    """Running taxonomy中的语义类别"""
    id: str
    name: str
    definition: str
    boundary_notes: str = ""
    positive_examples: List[str] = Field(default_factory=list)
    queries: List[Query] = Field(default_factory=list)

    def __str__(self):
        return f"Category(id={self.id}, name={self.name}, queries_count={len(self.queries)})"

    @property
    def query_count(self) -> int:
        return len(self.queries)

    @property
    def size(self) -> int:
        return len(self.queries)

    @property
    def samples(self) -> List[str]:
        return self.positive_examples

    @property
    def description(self) -> str:
        if self.boundary_notes:
            return f"{self.definition}\n边界说明：{self.boundary_notes}"
        return self.definition


class GraphState(TypedDict):
    """LangGraph流程的全局状态容器"""
    tasks: Deque[SplitTask]
    categories: Dict[str, Category]
    clusters_list: Optional[List[MicroCluster]]
    current_cluster: Optional[MicroCluster]
    residual_clusters: List[MicroCluster]
    total_queries: int
    min_cluster_size: int
    max_split_depth: int
