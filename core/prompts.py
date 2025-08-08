from typing import List, Dict

def create_review_prompt(finalized_categories: List[Dict], clusters_to_review: List[Dict]) -> str:
    categories= ""
    if finalized_categories:
        categories = """
{}
""".format(
            "\n".join([
                f"    <id>{cat['id']}</id>\n    <description>{cat['description']}</description>"
                for cat in finalized_categories
            ])
        )

    clusters_to_review = """
  {}""".format(
        "\n".join([
            f"    <cluster id=\"{cluster['id']}\">
        <samples>{', '.join(cluster['samples'])}</samples>
    </cluster>"
            for cluster in clusters_to_review
        ])
    )

    prompt_template = f"""<prompt>
You are an expert data analyst. Your task is to review unlabeled query clusters and output your decisions.

<categories>
{categories}
</categories>

For each cluster below, you must analyze the queries in the cluster and choose one of three actions: create, assign, or subdivide. 

<clusters_to_review>
{clusters_to_review}
</clusters_to_review>

<action_description>
create action: 
- if the queries in the cluster can be described by a common theme and can not be assigned to any existing category, you should create a new category.
- you must provide a description for the new category.
- 如果cluster属于同一个主题，可以一次性把一组cluster创建成一个category
assign action: 
- if the queries in the cluster can be assigned to an existing category, you should assign the cluster to the existing category.
- you must provide the id of the category to assign the cluster to.
subdivide action: 
- if the queries in the cluster can not be described by a common theme and can not be assigned to any existing category, you should subdivide the cluster into smaller clusters.
- you must provide the new k value for the K-Means clustering.
- the new k value should be a reasonable number, not too large or too small. 比如如果有2-3个类别的queries，k值可以设置为4-5个（为了能够分出目标类别）
</action_description>

Your entire response must be a single XML block enclosed by <output> and </output>.

<output>
<decision>
  <id>temp-a</id>
  <action>create</action>
  <description>询问订单物流和配送状态 e.g. 我的订单到哪了, 物流信息一直不更新, 快递什么时候派送</description>
</decision>
<decision>
  <id>temp-b</id>
  <action>assign</action>
  <target_id>CAT-001</target_id>
</decision>
<decision>
  <id>temp-c</id>
  <action>subdivide</action>
  <k_value>3</k_value>
</decision>
</decisions>
</output>
"""
    return prompt_template
