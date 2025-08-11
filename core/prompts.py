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
            f'    <cluster id="{cluster["id"]}">\n        <samples>{", ".join(cluster["samples"])}</samples>\n    </cluster>'
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
assign action: 
- if the queries in the cluster can be assigned to an existing category, you should assign the cluster to the existing category.
- you must provide the id of the category to assign the cluster to.
subdivide action: 
- if the queries in the cluster can not be described by a common theme and can not be assigned to any existing category, you should subdivide the cluster into smaller clusters.
- you must provide the new k value for the K-Means clustering.
</action_description>

<output_format_instructions>
Your entire response must be a single XML block with the following exact structure:
DO NOT wrap the XML in markdown code blocks (```xml or ```).
The XML must start directly with <output> and end with </output>:

<output>
  <decision>
    <id>cluster_id_here</id>
    <action>action_type_here</action>
    <!-- Only include one of the following based on action type -->
    <description>new_category_description</description> <!-- for create -->
    <target_id>existing_category_id</target_id> <!-- for assign -->
    <k_value>subdivision_count</k_value> <!-- for subdivide -->
  </decision>
  <!-- Include one decision block per cluster -->
</output>
</output_format_instructions>

<example_output>
<output>
  <decision>
    <id>cluster1</id>
    <action>create</action>
    <description>询问订单物流状态</description>
  </decision>
  <decision>
    <id>cluster2</id>
    <action>assign</action>
    <target_id>CAT-001</target_id>
  </decision>
</output>
</example_output>
"""
    return prompt_template
