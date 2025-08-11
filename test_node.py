from core.graph import TDRClusterGraph
from core.state import Query, Task, Cluster, Category
from collections import deque
import uuid
# from .state import GraphState, Category, Task, Query, Cluster  # Added Cluster to imports
import numpy as np
# import pytest

def test_clusterer_node():
    """测试clusterer节点功能"""
    graph = TDRClusterGraph()
    
    # 生成20个随机嵌入向量 (假设每个嵌入向量是384维)
    np.random.seed(42)  # 设置随机种子保证可重复性
    embeddings = np.random.rand(20, 384).tolist()
    
    # 准备测试数据
    queries = [
        Query(id=f"q{i}", content=f"测试查询{i}", embedding=embeddings[i])
        for i in range(20)
    ]
    
    initial_state = {
        "tasks": deque([Task(queries=queries, k_value=5)]),  # 分成5个簇
        "categories": {},
        "clusters_list": []
    }
    
    # 执行节点
    result = graph.clusterer_node(initial_state)
    
    # 验证结果
    assert len(result["clusters_list"]) == 5  # 应该分成5个簇
    assert not initial_state["tasks"]  # 任务队列应该被消费

def test_reviewer_node():
    """测试reviewer节点功能"""
    graph = TDRClusterGraph()
    
    # 准备测试数据 - 两个簇各10条语义不同的查询
    cluster1_queries = [
        Query(id=f"q1_{i}", content=content, embedding=[0.1 + i*0.01, 0.2 + i*0.01])
        for i, content in enumerate([
            "我的订单在哪里查看",
            "如何查询物流信息",
            "订单号123456的物流状态",
            "快递到哪了",
            "什么时候能收到货",
            "配送进度查询",
            "物流信息不更新怎么办",
            "预计送达时间",
            "快递员联系方式",
            "可以改收货地址吗"
        ])
    ]
    
    cluster2_queries = [
        Query(id=f"q2_{i}", content=content, embedding=[0.2 + i*0.01, 0.3 + i*0.01])
        for i, content in enumerate([
            "这件商品多少钱",
            "现在有优惠活动吗",
            "会员有什么折扣",
            "价格能便宜点吗",
            "双十一会打折吗",
            "买两件能优惠吗",
            "新人优惠券怎么领",
            "满减活动详情",
            "历史价格查询",
            "什么时候降价"
        ])
    ]
    
    clusters = [
        Cluster(id="c1", queries=cluster1_queries),
        Cluster(id="c2", queries=cluster2_queries)
    ]
    
    initial_state = {
        "tasks": deque(),
        "categories": {},
        "clusters_list": clusters
    }
    
    # 执行节点
    result = graph.reviewer_node(initial_state)
    
    # 验证结果
    assert len(result["clusters_list"]) == 2
    for cluster in result["clusters_list"]:
        assert cluster.judge is not None  # 每个簇应该有判断结果

def test_dispatcher_node():
    """测试dispatcher节点功能"""
    graph = TDRClusterGraph()
    
    # 准备测试数据 - 增加更多簇和不同action类型
    clusters = [
        # 创建新类别的簇
        Cluster(id="c1", queries=[Query(id="q1", content="查询订单", embedding=[0.1, 0.2])], 
                judge={"action": "create", "description": "订单查询"}),
        # 分配到现有类别的簇
        Cluster(id="c2", queries=[Query(id="q2", content="订单状态", embedding=[0.2, 0.3])],
                judge={"action": "assign", "target_id": "CAT-001"}),
        # 需要细分的簇
        Cluster(id="c3", queries=[
            Query(id="q3", content="商品价格", embedding=[0.3, 0.4]),
            Query(id="q4", content="促销活动", embedding=[0.4, 0.5])
        ], judge={"action": "subdivide", "k_value": 2}),
        # 另一个创建新类别的簇
        Cluster(id="c4", queries=[
            Query(id="q5", content="退货流程", embedding=[0.5, 0.6]),
            Query(id="q6", content="退款申请", embedding=[0.6, 0.7])
        ], judge={"action": "create", "description": "售后问题"}),
        # 另一个分配到现有类别的簇
        Cluster(id="c5", queries=[Query(id="q7", content="物流查询", embedding=[0.7, 0.8])],
                judge={"action": "assign", "target_id": "CAT-001"})
    ]
    
    initial_state = {
        "tasks": deque(),
        "categories": {
            "CAT-001": Category(id="CAT-001", description="订单", queries=[], samples=[]),
            "CAT-002": Category(id="CAT-002", description="物流", queries=[], samples=[])
        },
        "clusters_list": clusters
    }
    
    # 执行节点
    result = graph.dispatcher_node(initial_state)
    
    # 验证结果
    assert len(result["categories"]) == 4  # 应该有两个新创建的类别(订单查询和售后问题)
    assert not result["clusters_list"]  
    print (result["categories"])# 处理后的簇列表应该为空
    # 可以添加更多assert来验证每个action的处理结果

def test_should_continue():
    """测试should_continue条件判断"""
    graph = TDRClusterGraph()
    
    # 测试有任务的情况
    state_with_tasks = {
        "tasks": deque([Task(queries=[], k_value=2)]),
        "categories": {},
        "clusters_list": []
    }
    assert graph.should_continue(state_with_tasks) == "continue"
    
    # 测试无任务的情况
    state_no_tasks = {
        "tasks": deque(),
        "categories": {},
        "clusters_list": []
    }
    assert graph.should_continue(state_no_tasks) == "end"

test_dispatcher_node()
