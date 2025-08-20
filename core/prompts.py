"""
Prompt模板模块
包含LLM交互的所有Prompt模板，支持最新的多cluster创建单category功能
现在包含LLM调用功能
"""
from typing import List, Dict, Any
from .state import Category, Cluster


def create_review_prompt(finalized_categories: List[Dict[str, Any]], 
                        clusters_to_review: List[Dict[str, Any]],
                        dataset_name: str = None) -> str:
    """
    生成第一轮LLM决策的Prompt模板
    
    Args:
        finalized_categories: 已确定的类别列表（包含Rich Description）
        clusters_to_review: 待评审的clusters列表
        dataset_name: 数据集名称，用于获取特定的high-level目标
        
    Returns:
        str: 完整的prompt字符串
    """
    
    # 获取high-level目标配置
    from config.config_loader import CONFIG
    runtime_config = CONFIG.get('runtime', {})
    
    # 优先使用runtime中的high_level_goal，然后使用默认目标
    high_level_goal = runtime_config.get('high_level_goal')
    if not high_level_goal:
        high_level_goal = "对用户查询进行智能意图分类，生成具有明确业务含义的高质量类别"
    
    # 获取目标类别范围
    target_range = runtime_config.get('target_category_range', '15')
    
    
    # 构建现有类别展示
    existing_categories_xml = ""
    if finalized_categories:
        existing_categories_xml = "<existing_categories>\n"
        for cat in finalized_categories:
            existing_categories_xml += f"""  <category>
    <id>{cat['id']}</id>
    <description>{cat['description']}</description>
    <query_count>{len(cat.get('queries', []))}</query_count>
  </category>\n"""
        existing_categories_xml += "</existing_categories>\n\n"
    else:
        existing_categories_xml = "<existing_categories>\n  <!-- 暂无已确定的类别 -->\n</existing_categories>\n\n"
    
    # 构建待评审clusters展示
    clusters_xml = "<clusters_to_review>\n"
    for cluster in clusters_to_review:
        # 获取samples，直接使用clustering阶段已抽取的样本，避免重复抽样
        samples = cluster.get('samples', [])  # 使用clustering_service已抽取的样本
        samples_str = ', '.join([sample[:50] + ('...' if len(sample) > 50 else '') for sample in samples])
        
        query_count = len(cluster.get('queries', []))
        
        clusters_xml += f"""  <cluster id="{cluster['id']}">
    <samples>{samples_str}</samples>
    <query_count>{query_count}</query_count>
  </cluster>\n"""
    clusters_xml += "</clusters_to_review>"
    
    # 构建完整prompt
    prompt = f"""<role>You are an expert data analyst. Your task is to review unlabeled query clusters and output your decisions in a structured XML format. You must ensure EVERY cluster is judged exactly once.</role>

<high_level_goal>
{high_level_goal}

Based on this objective, analyze each cluster's semantic content thoughtfully and make intelligent classification decisions. Ensure every cluster is appropriately categorized to achieve comprehensive, balanced, and meaningful classification across the entire dataset. Pay careful attention to the semantic hierarchy, consistency, and logical coherence of the overall category structure.

**Target Category Count**: Aim for {target_range} final categories in total. Keep this range in mind when making decisions to achieve optimal classification granularity, only the subdivide action can increase the number of categories.
</high_level_goal>

{existing_categories_xml}<task>
  <instruction>
    For each cluster below, choose ONE action: create, assign, or subdivide.
    
    **Decision Rules:**
    - **create**: Create new meaningful category
      - Support multiple clusters: "cluster-a,cluster-b" format if multiple clusters are related to a same category, use "cluster-a,cluster-b" format
      - Use when you confirm that the cluster cannot be assigned to any existing category AND it is internally semantically coherent. 
      - Provide Rich Description with specific examples
      - Target: ONE new category
    - **assign**: Enhance existing category by adding related content
      - Support multiple clusters: "cluster-a,cluster-b" format, if multiple clusters are related to the same category, use "cluster-a,cluster-b" format
      - Use when cluster content is related to an existing category and can refine/expand its scope
      - Must specify target_id (ONE existing category) and description_update
      - For description_update, choose "no_update" for perfect matches, or provide refined description for semantic expansion
    - **subdivide**: Split cluster into smaller clusters  
      - Single cluster only
      - Use when cluster contains multiple categories that can be separated
      - Choose k_value wisely: should be larger than the number of themes you identify, giving the algorithm room to separate them effectively
    
    **CRITICAL**: Every cluster must appear in exactly one decision. No cluster should be missed or duplicated.
  </instruction>
  
  {clusters_xml}
</task>

<format_requirements>
  Your entire response must be a single XML block with <decisions> as the root element:
  
  <decisions>
    <decision>
      <id>cluster-id(s)</id>
      <action>create|assign|subdivide</action>
      <!-- For CREATE: provide description -->
      <!-- For ASSIGN: provide target_id and description_update -->
      <!-- For SUBDIVIDE: provide k_value -->
    </decision>
    <!-- More decisions... -->
  </decisions>
  
  For CREATE actions, provide Rich Description with this format:
  "主要描述 - 详细解释和覆盖范围\\n典型例子：具体例子1、例子2、例子3"
</format_requirements>

<example>
  <decisions>
    <decision>
      <id>temp-c,temp-d</id>
      <action>create</action>
      <description>用户寻求客户服务支持 - 包括联系客服、退货退款、问题解决等服务请求
      典型例子：怎么联系客服、我要退货、退款多久到账、怎么投诉</description>
    </decision>
    <decision>
      <id>temp-a,temp-e</id>
      <action>assign</action>
      <target_id>CAT-001</target_id>
      <description_update>用户关于订单和物流的全程关注 - 涵盖发货通知、物流追踪、配送状态、时效查询等订单履约全链路问题
      典型例子：什么时候发货、我的订单到哪了、快递什么时候到、为什么物流信息不更新、预计多久能到</description_update>
    </decision>
    <decision>
      <id>temp-b</id>
      <action>subdivide</action>
      <k_value>4</k_value>
    </decision>
    <decision>
      <id>temp-f</id>
      <action>subdivide</action>
      <k_value>3</k_value>
    </decision>
  </decisions>
</example>

Please provide your decisions now in the required <decisions> XML format:"""

    return prompt



def create_category_display_dict(category: Category) -> Dict[str, Any]:
    """
    将Category对象转换为展示用的字典格式
    
    Args:
        category: Category对象
        
    Returns:
        Dict[str, Any]: 用于展示的字典
    """
    return {
        'id': category.id,
        'description': category.description,
        'queries': category.queries,
        'query_count': len(category.queries),
        'samples': category.samples
    }


def create_cluster_display_dict(cluster: Cluster) -> Dict[str, Any]:
    """
    将Cluster对象转换为展示用的字典格式
    
    Args:
        cluster: Cluster对象
        
    Returns:
        Dict[str, Any]: 用于展示的字典
    """
    return {
        'id': cluster.id,
        'queries': cluster.queries,
        'samples': cluster.samples,
        'query_count': len(cluster.queries)
    }






