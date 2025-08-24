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
    Your goal is to categorize the clusters below. For each cluster, you must choose ONE of three actions: `assign`, `subdivide`, or `create`.

    **Core Principle: Granularity is Key. Focus on Specific, Actionable User Intents.**
    Your primary goal is to define categories that represent a *single, distinct, and actionable user intent*. Do not group queries by broad topics. A shared keyword (e.g., 'credit card') is insufficient. The queries must share a common, specific goal (e.g., 'apply for a credit card,' not 'credit card information').

    **Bad, Topic-Based Categories (AVOID THESE):**
    - "Credit Card Issues"
    - "Account Information"
    - "Technical Problems"
    - "Product Questions"

    **Your Decision-Making Flow (Internal Thoughts - Do NOT output this):**

    1.  **Analyze the Cluster**: What is the core, specific *goal* of the users in this cluster? Is there only one goal or multiple?

    2.  **PRIORITY 1: `assign`**: 
        - **Question**: Does this cluster's single, specific intent perfectly match the intent of an *existing* category?
        - **Action**: If yes, `assign` it. If the cluster's queries can enrich the existing category's description, provide a `description_update`.

    3.  **PRIORITY 2: `subdivide`**:
        - **Question**: Does this cluster contain multiple distinct user intents? Or does it represent a broad topic that needs to be broken down into specific intents?
        - **Action**: If yes, `subdivide`. Determine the exact number of unique, specific intents (`k_value`) you can identify.

    4.  **PRIORITY 3: `create`**:
        - **Question**: Does this cluster represent a *single, cohesive, and new* user intent that doesn't fit any existing category?
        - **Action**: If yes, and only then, `create` a new category. The description must be highly specific.

    **Use Query Count as a Heuristic:**
    The `query_count` for each cluster is a valuable signal for your decision-making.
    - **High Query Count (e.g., >100)**: These clusters are often too broad. Be very cautious with `create`. prefer `subdivide` and use a higher `k_value` to break it down into more granular intents. A large cluster is unlikely to represent a single new intent.
    - **Low Query Count (e.g., <20)**: These clusters are more likely to represent a niche, specific intent. They are good candidates for `create` if they are cohesive and don't fit existing categories, or for `assign` if they match one.

    **Action Rules (For your final XML output):**
    - **`assign`**: Provide `target_id`. Also provide `description_update` if you are refining the category description; otherwise, use `no_update`.
    - **`subdivide`**: Provide a `k_value` (integer from 2 to 5) representing the number of distinct intents you've identified.
    - **`create`**: Provide a rich `description` for the new category, following the required format.
    
    **CRITICAL**: Every cluster must be judged exactly once. Your final output must be a single `<decisions>` XML block as shown in the example.
  </instruction>
  
  {clusters_xml}
</task>

<good_vs_bad_examples>
  To help you understand the required granularity, here are examples:

  **Example 1: A cluster about 'Credit Cards'**
  - **BAD DECISION**: `create` a single category named "Credit Card Issues". This is too broad.
  - **GOOD DECISION**: `subdivide` the cluster into multiple, specific intent-based categories like:
    - Category A: "Credit Card Application & Status" (queries: 'how to apply', 'check my application status')
    - Category B: "Credit Card Billing & Payments" (queries: 'when is my bill due', 'how to pay my bill')
    - Category C: "Reporting Lost or Stolen Cards" (queries: 'I lost my card', 'someone stole my wallet')

  **Example 2: A cluster about 'Account Information'**
  - **BAD DECISION**: `create` a single category named "Account Info".
  - **GOOD DECISION**: `subdivide` it into:
    - Category A: "Checking Account Balance" (queries: 'what's my balance', 'how much money do I have')
    - Category B: "Updating Personal Information" (queries: 'change my address', 'update my phone number')
</good_vs_bad_examples>

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