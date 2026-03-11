"""
Prompt模板模块
包含LLM交互的所有Prompt模板
"""
from typing import List, Dict, Any, Optional
from xml.sax.saxutils import escape

from .state import Category, MicroCluster


def create_reviewer_system_prompt() -> str:
    """生成reviewer的系统提示"""
    return """You maintain a running intent taxonomy over semantically coherent micro-clusters.

Core principles:
- Prioritize semantic correctness over any preset target number of categories.
- Produce flat leaf-level categories in this stage. Do not invent parent-child hierarchy here.
- Keep category boundaries stable, reusable, and easy for humans to understand.
- Avoid unnecessary fragmentation. Prefer merge when an existing category can absorb the new evidence without becoming vague.
- Use create only when the cluster expresses a genuinely new reusable intent.
- Use split only when the cluster still clearly mixes multiple distinct intents and should be sent back for further physical splitting.
- The evidence may be grouped into representative, diverse, and boundary samples. Use all groups jointly when judging both purity and category fit.
- Category names should be concise, domain-faithful, and reusable.
- If a merge broadens the meaning of an existing category, update the category name only when the current name becomes too narrow or misleading.
- Follow the dominant language of the existing taxonomy and cluster samples when writing names, definitions, and boundary notes. If the taxonomy already has a dominant language, stay consistent with it.
- Return exactly one XML <decisions> block and nothing else."""


def create_review_prompt(
    finalized_categories: List[Dict[str, Any]],
    cluster_to_review: Dict[str, Any],
) -> str:
    """生成V2 reviewer使用的用户提示"""
    from config.config_loader import CONFIG

    runtime_config = CONFIG.get("runtime", {})
    taxonomy_context = runtime_config.get("taxonomy_context")

    context_xml = _create_context_block(taxonomy_context)
    existing_categories_xml = _create_existing_categories_block(finalized_categories)
    cluster_xml = _create_cluster_block(cluster_to_review)

    return f"""{context_xml}{existing_categories_xml}
<task>
  <instruction>
    Review the single micro-cluster below and choose exactly ONE action: create, merge, or split.

    Decision rules:
    - Read the evidence holistically:
      representative samples show the semantic center,
      diverse samples show major internal variation,
      boundary samples show hard-edge or suspicious cases most likely to reveal impurity.

    - create:
      Use when this micro-cluster expresses a new intent that is not covered by the current taxonomy.
      Only create when the intent is stable and reusable beyond these specific examples.
      You must provide:
      - name
      - definition
      - boundary_notes

    - merge:
      Use when this micro-cluster belongs to an existing category.
      Prefer merge when the existing category can absorb the cluster without losing clarity.
      You must provide:
      - target_id
      - name_update
      - definition_update
      - boundary_update
      Use "no_update" for any field that does not need revision.
      Only update the name when the existing name has become too narrow or inaccurate after the merge.

    - split:
      Use when this micro-cluster is still semantically mixed and should be sent back for another round of physical splitting.
      Do not use split merely to chase a finer taxonomy.
      You must provide:
      - reason

    Hard constraints:
    - Review exactly one micro-cluster in this call.
    - Return exactly one <decision>.
    - Each decision must reference exactly one micro-cluster id.
    - Do not output noise decisions. Noise is handled by the system.
  </instruction>

  {cluster_xml}
</task>

<format_requirements>
  Your response must be a single XML block with <decisions> as the root element.
  It must contain exactly one <decision>.

  <decisions>
    <decision>
      <id>micro-cluster-id</id>
      <action>create|merge|split</action>
      <!-- CREATE -->
      <name>category name</name>
      <definition>semantic definition</definition>
      <boundary_notes>what belongs here and what does not</boundary_notes>
      <!-- MERGE -->
      <target_id>CAT-001</target_id>
      <name_update>no_update or refined category name</name_update>
      <definition_update>no_update or refined definition</definition_update>
      <boundary_update>no_update or refined boundary notes</boundary_update>
      <!-- SPLIT -->
      <reason>why this cluster still mixes multiple intents</reason>
    </decision>
  </decisions>
</format_requirements>

<example>
  <decisions>
    <decision>
      <id>mc-001</id>
      <action>merge</action>
      <target_id>CAT-003</target_id>
      <name_update>no_update</name_update>
      <definition_update>no_update</definition_update>
      <boundary_update>Extend the boundary to include requests about changing the delivery address after an order is placed.</boundary_update>
    </decision>
  </decisions>
</example>

Please provide your decision now in the required XML format:"""


def _create_context_block(taxonomy_context: Optional[str]) -> str:
    if not taxonomy_context:
        return ""

    return f"""<taxonomy_context>
{_xml_text(taxonomy_context)}
</taxonomy_context>

"""


def _create_existing_categories_block(finalized_categories: List[Dict[str, Any]]) -> str:
    existing_categories_xml = "<existing_categories>\n"
    if finalized_categories:
        for cat in finalized_categories:
            positive_examples = " | ".join(cat.get("positive_examples", [])[:3]) or "none"
            existing_categories_xml += f"""  <category>
    <id>{_xml_text(cat['id'])}</id>
    <name>{_xml_text(cat['name'])}</name>
    <definition>{_xml_text(cat['definition'])}</definition>
    <boundary_notes>{_xml_text(cat['boundary_notes'] or 'none')}</boundary_notes>
    <query_count>{cat['query_count']}</query_count>
    <positive_examples>{_xml_text(positive_examples)}</positive_examples>
  </category>\n"""
    else:
        existing_categories_xml += "  <!-- No finalized categories yet -->\n"
    existing_categories_xml += "</existing_categories>"
    return existing_categories_xml


def _create_cluster_block(cluster: Dict[str, Any]) -> str:
    evidence_rounds_xml = _create_evidence_rounds_block(cluster.get("evidence_rounds", []))
    return f"""<micro_cluster_to_review>
  <cluster>
    <id>{_xml_text(cluster['id'])}</id>
    <depth>{cluster['depth']}</depth>
    <query_count>{cluster['query_count']}</query_count>
  </cluster>
{evidence_rounds_xml}
</micro_cluster_to_review>"""


def _create_evidence_rounds_block(evidence_rounds: List[Dict[str, Any]]) -> str:
    if not evidence_rounds:
        return "  <evidence_rounds>\n    <evidence_round mode=\"initial\" round=\"1\">\n      <note>none</note>\n    </evidence_round>\n  </evidence_rounds>"

    lines = ["  <evidence_rounds>"]
    for evidence_round in evidence_rounds:
        lines.append(
            f"    <evidence_round mode=\"{_xml_text(evidence_round.get('mode', 'initial'))}\" "
            f"round=\"{int(evidence_round.get('round', 1))}\">"
        )
        lines.append(f"      <sample_count>{int(evidence_round.get('sample_count', 0))}</sample_count>")
        groups = evidence_round.get("groups", {})
        for group_name, samples in groups.items():
            lines.append(f"      <{group_name}_samples>")
            for sample in samples:
                lines.append(f"        <sample>{_xml_text(sample)}</sample>")
            lines.append(f"      </{group_name}_samples>")
        lines.append("    </evidence_round>")
    lines.append("  </evidence_rounds>")
    return "\n".join(lines)


def _xml_text(value: Any) -> str:
    return escape(str(value))


def create_category_display_dict(category: Category) -> Dict[str, Any]:
    """将Category对象转换为展示用的字典格式"""
    return {
        "id": category.id,
        "name": category.name,
        "definition": category.definition,
        "boundary_notes": category.boundary_notes,
        "query_count": len(category.queries),
        "positive_examples": category.positive_examples,
    }


def create_cluster_display_dict(cluster: MicroCluster) -> Dict[str, Any]:
    """将MicroCluster对象转换为展示用的字典格式"""
    return {
        "id": cluster.id,
        "depth": cluster.depth,
        "query_count": len(cluster.queries),
        "sample_texts": cluster.sample_texts,
        "evidence_rounds": cluster.evidence_rounds,
    }
