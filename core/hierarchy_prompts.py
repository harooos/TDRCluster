"""
Hierarchy prompt templates.
"""
from typing import Dict, Any, List
from xml.sax.saxutils import escape


def create_hierarchy_system_prompt() -> str:
    return """You organize an existing flat intent taxonomy into a browsing hierarchy.

Core rules:
- Do not invent, rename, split, merge, or drop leaf categories.
- Every leaf category id must appear exactly once in the hierarchy.
- Create internal nodes only when they add real abstraction value.
- Use as many levels as needed, but no more than necessary.
- Avoid single-child chains unless they are semantically unavoidable.
- Keep sibling nodes at similar semantic granularity.
- Classify by the leaf category's primary user intent, not by surface keywords alone.
- An internal node must be a truthful abstraction for every child under it.
- If a leaf only weakly fits a subgroup, place it higher in the tree instead of forcing it into that subgroup.
- Internal node names should be concise, reusable, and informative.
- Internal node descriptions should explain the common abstraction shared by their children.
- If a leaf does not fit any meaningful subgroup, it may appear directly under a higher-level node.
- Return JSON only. No markdown fences. No extra commentary."""


def create_hierarchy_prompt(
    categories: List[Dict[str, Any]],
    dataset_name: str,
) -> str:
    category_xml = _create_category_block(categories)
    return f"""<task>
  <goal>
    Build a semantic hierarchy over the flat leaf taxonomy for dataset "{escape(dataset_name)}".
    The hierarchy is for human browsing at multiple semantic granularities.
  </goal>

  <instructions>
    1. Preserve all leaf categories exactly as provided.
    2. Create internal grouping nodes only when they have a clear shared abstraction.
    3. The tree depth should emerge naturally from the leaf taxonomy.
    4. Do not force every branch to have the same depth.
    5. Use the primary semantic intent of each leaf rather than shallow lexical overlap.
    6. If a subgroup is semantically weak, keep the leaf directly under a higher node rather than forcing a dubious subgroup.
    7. Avoid empty abstractions such as "General issues" unless the children truly share no better unifying concept.
    8. Prefer a compact, meaningful hierarchy over a shallow but noisy one.
  </instructions>

  <output_contract>
    Return one JSON object with exactly this top-level shape:
    {{
      "roots": [NODE, NODE, ...]
    }}

    A NODE must be exactly one of:

    1. Internal node:
    {{
      "name": "internal node name",
      "description": "shared abstraction of all children",
      "children": [NODE, NODE, ...]
    }}

    2. Leaf node:
    {{
      "leaf_id": "CAT-001"
    }}

    Constraints:
    - Every provided leaf_id must appear exactly once.
    - Never include any unknown leaf_id.
    - An internal node must have at least 2 children unless absolutely unavoidable.
    - Do not add fields beyond those defined above.
  </output_contract>

{category_xml}
</task>"""


def create_retry_suffix(error_message: str) -> str:
    return f"""

<validation_error>
The previous output was invalid:
{escape(error_message)}
</validation_error>

Please correct the hierarchy and return valid JSON only."""


def _create_category_block(categories: List[Dict[str, Any]]) -> str:
    lines = ["<leaf_categories>"]
    for category in categories:
        positive_examples = " | ".join(
            _truncate_text(example, 80)
            for example in category.get("positive_examples", [])[:2]
        ) or "none"
        lines.append("  <leaf_category>")
        lines.append(f"    <id>{escape(category['id'])}</id>")
        lines.append(f"    <name>{escape(category['name'])}</name>")
        lines.append(
            f"    <definition>{escape(_truncate_text(category['definition'], 240))}</definition>"
        )
        boundary_notes = category.get("boundary_notes") or "none"
        lines.append(f"    <boundary_notes>{escape(_truncate_text(boundary_notes, 180))}</boundary_notes>")
        lines.append(f"    <query_count>{int(category['query_count'])}</query_count>")
        lines.append(f"    <positive_examples>{escape(positive_examples)}</positive_examples>")
        lines.append("  </leaf_category>")
    lines.append("</leaf_categories>")
    return "\n".join(lines)


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3].rstrip()}..."
