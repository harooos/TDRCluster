"""
Artifact saving and single-file agent report generation for IntentAtlas.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REPORT_TEXT = {
    "zh": {
        "title": "# IntentAtlas Agent Report: {display_name}",
        "run_overview": "## 运行概览",
        "dataset": "- 数据集：`{display_name}`",
        "dataset_id": "- 内部数据集 ID：`{dataset}`",
        "timestamp": "- 生成时间：`{timestamp}`",
        "mode": "- 运行模式：`{mode}`",
        "runtime": "- 运行耗时（秒）：`{runtime_seconds}`",
        "llm_model": "- LLM 模型：`{llm_model}`",
        "embedding_model": "- Embedding 模型：`{embedding_model}`",
        "total_queries": "- 总 query 数：`{total_queries}`",
        "classified_queries": "- 已归类 query 数：`{classified_query_count}`",
        "residual_queries": "- 残差 query 数：`{residual_query_count}`",
        "leaf_categories": "- 叶子类别数：`{total_categories}`",
        "residual_clusters": "- 残差簇数：`{residual_cluster_count}`",
        "insight_summary": "## AI 解读摘要",
        "summary_top": "- 主题地图最终收敛为 `{total_categories}` 个叶子主题。当前规模最大的主题包括 {top_names}。",
        "summary_residual": "- 残差占比为 `{residual_ratio}`。这个值越低，通常意味着更多 query 被稳定吸收到主题地图中。",
        "summary_long_tail": "- 当前报告中有 `{long_tail_count}` 个长尾类别，其 query 数不超过 5。",
        "summary_residual_hint": "- 残差簇值得重点查看，它们可能对应新意图、脏数据或仍然模糊的边界。",
        "hierarchy_view": "## 层级视图",
        "no_hierarchy": "- 当前运行未生成层级树。",
        "leaf_taxonomy": "## 主题清单",
        "category_heading": "### {name}",
        "query_count": "- Query 数：`{query_count}`",
        "definition": "- 定义：{definition}",
        "boundary_notes": "- 边界说明：{boundary_notes}",
        "representative_examples": "- 代表样本：",
        "residual_area": "## 残差与模糊区域",
        "residual_heading": "### 待确认分组 {index}",
        "reason": "- 原因：{reason}",
        "sample_texts": "- 样本文本：",
        "no_residual": "- 当前运行没有记录残差簇。",
        "suggested_actions": "## 建议动作",
        "largest_categories": "- 建议优先查看规模最大的几个主题：{category_names}。",
        "residual_volume_hint": "- 当前残差体量不小，建议在业务落地前先人工查看残差区域。",
        "long_tail_hint": "- 长尾类别可能对应小众但真实的问题，建议判断它们应该合并、监控还是单独处理。",
        "chatbot_hint": "- 这份报告适合直接拖入 chatbot，继续做更深入的业务分析或问答。",
        "follow_up_questions": "## 建议继续追问大模型的问题",
        "follow_up_1": "- 哪些类别从业务视角应该合并，即使它们在语义上可以独立成类？",
        "follow_up_2": "- 哪些残差簇更像是新意图，哪些更像是数据质量问题或噪声？",
        "follow_up_3": "- 哪些头部类别适合再归纳为更高层的周报或月报主题？",
        "follow_up_4": "- 哪些长尾类别规模虽小，但值得单独监控或建立 FAQ？",
        "curation_log": "## 人工修订记录",
        "no_curation": "- 当前没有记录人工修订操作。",
        "structured_appendix": "## 结构化附录",
    },
    "en": {
        "title": "# IntentAtlas Agent Report: {display_name}",
        "run_overview": "## Run Overview",
        "dataset": "- Dataset: `{display_name}`",
        "dataset_id": "- Internal dataset id: `{dataset}`",
        "timestamp": "- Timestamp: `{timestamp}`",
        "mode": "- Mode: `{mode}`",
        "runtime": "- Runtime seconds: `{runtime_seconds}`",
        "llm_model": "- LLM model: `{llm_model}`",
        "embedding_model": "- Embedding model: `{embedding_model}`",
        "total_queries": "- Total queries: `{total_queries}`",
        "classified_queries": "- Classified queries: `{classified_query_count}`",
        "residual_queries": "- Residual queries: `{residual_query_count}`",
        "leaf_categories": "- Leaf categories: `{total_categories}`",
        "residual_clusters": "- Residual clusters: `{residual_cluster_count}`",
        "insight_summary": "## AI Narrative Summary",
        "summary_top": "- The intent map converged to `{total_categories}` leaf themes. The largest visible themes are {top_names}.",
        "summary_residual": "- Residual coverage is `{residual_ratio}` of total queries. Lower usually means more queries were absorbed into the intent map.",
        "summary_long_tail": "- There are `{long_tail_count}` long-tail categories in this report excerpt with at most 5 queries.",
        "summary_residual_hint": "- Residual clusters deserve review because they may indicate new intents, noisy logs, or unresolved ambiguity.",
        "hierarchy_view": "## Hierarchy View",
        "no_hierarchy": "- No hierarchy artifact is available for this run.",
        "leaf_taxonomy": "## Themes",
        "category_heading": "### {name}",
        "query_count": "- Query count: `{query_count}`",
        "definition": "- Definition: {definition}",
        "boundary_notes": "- Boundary notes: {boundary_notes}",
        "representative_examples": "- Representative examples:",
        "residual_area": "## Residual And Ambiguous Areas",
        "residual_heading": "### Unresolved Group {index}",
        "reason": "- Reason: {reason}",
        "sample_texts": "- Sample texts:",
        "no_residual": "- No residual clusters were recorded in this run.",
        "suggested_actions": "## Suggested Actions",
        "largest_categories": "- Review the largest themes first: {category_names}.",
        "residual_volume_hint": "- Residual volume is non-trivial. Inspect the residual section before operationalizing the intent map.",
        "long_tail_hint": "- Long-tail categories may represent niche but actionable issues. Consider whether they should be merged, monitored, or escalated.",
        "chatbot_hint": "- This report is designed to be dropped into a chatbot for deeper analysis and business-facing writeups.",
        "follow_up_questions": "## Suggested Follow-up Questions For Another LLM",
        "follow_up_1": "- Which categories should be merged from a business perspective even if the semantic split is defensible?",
        "follow_up_2": "- Which residual clusters look like new intents, data quality problems, or logging noise?",
        "follow_up_3": "- Which top categories should be grouped into higher-level themes for weekly or monthly reporting?",
        "follow_up_4": "- Which long-tail categories are too small to operationalize, and which deserve separate monitoring?",
        "curation_log": "## Curation Log",
        "no_curation": "- No manual curation actions were recorded for this run.",
        "structured_appendix": "## Structured Appendix",
    },
}


class ReportService:
    """Build a single markdown report for humans and LLMs."""

    def save_agent_report(
        self,
        *,
        output_path: Path,
        summary: Dict[str, Any],
        hierarchy: Optional[Dict[str, Any]],
        run_metadata: Dict[str, Any],
        language: str = "zh",
        curation_log: Optional[List[Dict[str, Any]]] = None,
        llm_service: Optional[Any] = None,
        enable_llm_summary: bool = True,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.build_agent_report(
                summary=summary,
                hierarchy=hierarchy,
                run_metadata=run_metadata,
                language=language,
                curation_log=curation_log or [],
                llm_service=llm_service,
                enable_llm_summary=enable_llm_summary,
            ),
            encoding="utf-8",
        )
        return output_path

    def build_agent_report(
        self,
        *,
        summary: Dict[str, Any],
        hierarchy: Optional[Dict[str, Any]] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        language: str = "zh",
        curation_log: Optional[List[Dict[str, Any]]] = None,
        llm_service: Optional[Any] = None,
        enable_llm_summary: bool = True,
    ) -> str:
        run_metadata = run_metadata or {}
        language = language if language in REPORT_TEXT else "zh"
        text = REPORT_TEXT[language]
        categories = self._ordered_categories(summary)
        residuals = self._ordered_residuals(summary)
        run_mode = run_metadata.get("run_mode", summary.get("mode", "unknown"))
        residual_ratio = summary.get("residual_query_count", 0) / max(
            1,
            summary.get("total_queries", summary.get("classified_query_count", 0) + summary.get("residual_query_count", 0)),
        )
        top_categories = categories[:8]
        long_tail_categories = [category for category in categories if category["query_count"] <= 5][:10]
        display_name = run_metadata.get("display_name") or summary.get("display_name") or summary.get("dataset", "unknown")
        curation_log = curation_log or summary.get("curation_log", [])
        llm_summary = self._generate_llm_summary(
            summary=summary,
            hierarchy=hierarchy,
            run_metadata=run_metadata,
            language=language,
            llm_service=llm_service,
            enable_llm_summary=enable_llm_summary,
        )

        lines: list[str] = []
        lines.append(text["title"].format(display_name=display_name))
        lines.append("")
        lines.append(text["run_overview"])
        lines.append("")
        lines.append(text["dataset"].format(display_name=display_name))
        lines.append(text["dataset_id"].format(dataset=summary.get("dataset", "unknown")))
        lines.append(text["timestamp"].format(timestamp=summary.get("timestamp", "unknown")))
        lines.append(text["mode"].format(mode=run_mode))
        lines.append(text["runtime"].format(runtime_seconds=run_metadata.get("runtime_seconds", "unknown")))
        lines.append(text["llm_model"].format(llm_model=run_metadata.get("llm_model", "unknown")))
        lines.append(text["embedding_model"].format(embedding_model=run_metadata.get("embedding_model", "unknown")))
        lines.append(text["total_queries"].format(total_queries=summary.get("total_queries", summary.get("classified_query_count", 0))))
        lines.append(text["classified_queries"].format(classified_query_count=summary.get("classified_query_count", summary.get("total_queries", 0))))
        lines.append(text["residual_queries"].format(residual_query_count=summary.get("residual_query_count", 0)))
        lines.append(text["leaf_categories"].format(total_categories=summary.get("total_categories", 0)))
        lines.append(text["residual_clusters"].format(residual_cluster_count=summary.get("residual_cluster_count", 0)))
        lines.append("")

        lines.append(text["insight_summary"])
        lines.append("")
        if llm_summary:
            lines.append(llm_summary)
        else:
            if top_categories:
                top_names = ", ".join(
                    f"{category['name']} ({category['query_count']})"
                    for category in top_categories[:3]
                )
                lines.append(text["summary_top"].format(
                    total_categories=summary.get("total_categories", 0),
                    top_names=top_names,
                ))
            lines.append(
                text["summary_residual"].format(residual_ratio=f"{residual_ratio:.2%}")
            )
            if long_tail_categories:
                lines.append(text["summary_long_tail"].format(long_tail_count=len(long_tail_categories)))
            if residuals:
                lines.append(text["summary_residual_hint"])
        lines.append("")

        lines.append(text["hierarchy_view"])
        lines.append("")
        if hierarchy and hierarchy.get("roots"):
            lines.extend(
                self._render_hierarchy_lines(
                    hierarchy["roots"],
                    category_name_map={category["id"]: category["name"] for category in categories},
                )
            )
        else:
            lines.append(text["no_hierarchy"])
        lines.append("")

        lines.append(text["leaf_taxonomy"])
        lines.append("")
        for category in categories:
            lines.append(text["category_heading"].format(name=category["name"]))
            lines.append(text["query_count"].format(query_count=category["query_count"]))
            lines.append(text["definition"].format(definition=category["definition"]))
            if category.get("boundary_notes"):
                lines.append(text["boundary_notes"].format(boundary_notes=category["boundary_notes"]))
            examples = category.get("positive_examples", [])
            if examples:
                lines.append(text["representative_examples"])
                for example in examples[:5]:
                    lines.append(f"  - {example}")
            lines.append("")

        lines.append(text["residual_area"])
        lines.append("")
        if residuals:
            for index, residual in enumerate(residuals[:10], 1):
                lines.append(text["residual_heading"].format(index=index))
                lines.append(text["query_count"].format(query_count=residual["query_count"]))
                lines.append(text["reason"].format(reason=residual["reason"]))
                if residual.get("sample_texts"):
                    lines.append(text["sample_texts"])
                    for sample in residual["sample_texts"][:5]:
                        lines.append(f"  - {sample}")
                lines.append("")
        else:
            lines.append(text["no_residual"])
        lines.append("")

        lines.append(text["suggested_actions"])
        lines.append("")
        if top_categories:
            lines.append(
                text["largest_categories"].format(
                    category_names=", ".join(category["name"] for category in top_categories[:5])
                )
            )
        if residual_ratio >= 0.1:
            lines.append(text["residual_volume_hint"])
        if long_tail_categories:
            lines.append(text["long_tail_hint"])
        lines.append(text["chatbot_hint"])
        lines.append("")

        lines.append(text["follow_up_questions"])
        lines.append("")
        lines.append(text["follow_up_1"])
        lines.append(text["follow_up_2"])
        lines.append(text["follow_up_3"])
        lines.append(text["follow_up_4"])
        lines.append("")

        lines.append(text["curation_log"])
        lines.append("")
        if curation_log:
            for item in curation_log:
                lines.append(f"- {item}")
        else:
            lines.append(text["no_curation"])
        lines.append("")

        lines.append(text["structured_appendix"])
        lines.append("")
        appendix = {
            "dataset": summary.get("dataset"),
            "display_name": display_name,
            "timestamp": summary.get("timestamp"),
            "mode": run_mode,
            "language": language,
            "total_queries": summary.get("total_queries"),
            "classified_query_count": summary.get("classified_query_count"),
            "residual_query_count": summary.get("residual_query_count"),
            "total_categories": summary.get("total_categories"),
            "residual_cluster_count": summary.get("residual_cluster_count"),
            "categories": summary.get("categories", {}),
            "residual_clusters": summary.get("residual_clusters", {}),
            "hierarchy": hierarchy or {},
            "run_metadata": run_metadata,
            "curation_log": curation_log,
            "report_metadata": {
                "llm_summary_enabled": enable_llm_summary,
                "llm_summary_generated": bool(llm_summary),
                "llm_summary_model": self._llm_model_name(llm_service) if llm_summary else None,
            },
        }
        lines.append("```json")
        lines.append(json.dumps(appendix, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    def load_summary(self, summary_path: str | Path) -> Dict[str, Any]:
        return json.loads(Path(summary_path).read_text(encoding="utf-8"))

    def load_hierarchy(self, hierarchy_path: str | Path) -> Dict[str, Any]:
        return json.loads(Path(hierarchy_path).read_text(encoding="utf-8"))

    def _ordered_categories(self, summary: Dict[str, Any]) -> list[Dict[str, Any]]:
        categories = []
        for category_id, category in summary.get("categories", {}).items():
            categories.append(
                {
                    "id": category_id,
                    **category,
                }
            )
        categories.sort(key=lambda item: (-item["query_count"], item["id"]))
        return categories

    def _ordered_residuals(self, summary: Dict[str, Any]) -> list[Dict[str, Any]]:
        residuals = []
        for cluster_id, cluster in summary.get("residual_clusters", {}).items():
            residuals.append(
                {
                    "id": cluster_id,
                    **cluster,
                }
            )
        residuals.sort(key=lambda item: (-item["query_count"], item["id"]))
        return residuals

    def _render_hierarchy_lines(
        self,
        roots: list[Dict[str, Any]],
        *,
        depth: int = 0,
        category_name_map: Optional[Dict[str, str]] = None,
    ) -> list[str]:
        lines: list[str] = []
        indent = "  " * depth
        category_name_map = category_name_map or {}
        for node in roots:
            if "leaf_id" in node:
                lines.append(f"{indent}- {category_name_map.get(node['leaf_id'], node['leaf_id'])}")
                continue
            lines.append(f"{indent}- {node['name']}: {node['description']}")
            lines.extend(
                self._render_hierarchy_lines(
                    node.get("children", []),
                    depth=depth + 1,
                    category_name_map=category_name_map,
                )
            )
        return lines

    def _generate_llm_summary(
        self,
        *,
        summary: Dict[str, Any],
        hierarchy: Optional[Dict[str, Any]],
        run_metadata: Dict[str, Any],
        language: str,
        llm_service: Optional[Any],
        enable_llm_summary: bool,
    ) -> Optional[str]:
        if not enable_llm_summary or llm_service is None:
            return None

        payload = self._build_llm_summary_payload(
            summary=summary,
            hierarchy=hierarchy,
            run_metadata=run_metadata,
        )
        system_prompt, user_prompt = self._build_llm_summary_prompt(payload, language=language)

        try:
            response = llm_service.simple_call(
                user_prompt,
                system_message=system_prompt,
                max_tokens=1400,
                timeout=90,
            )
        except Exception:
            return None

        cleaned = self._sanitize_markdown_block(response)
        return cleaned or None

    def _build_llm_summary_payload(
        self,
        *,
        summary: Dict[str, Any],
        hierarchy: Optional[Dict[str, Any]],
        run_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        categories = self._ordered_categories(summary)
        residuals = self._ordered_residuals(summary)
        total_queries = summary.get("total_queries", 0) or (
            summary.get("classified_query_count", 0) + summary.get("residual_query_count", 0)
        )
        residual_ratio = summary.get("residual_query_count", 0) / max(1, total_queries)

        return {
            "dataset": summary.get("dataset"),
            "display_name": run_metadata.get("display_name") or summary.get("display_name") or summary.get("dataset"),
            "timestamp": summary.get("timestamp"),
            "run_mode": run_metadata.get("run_mode", summary.get("mode", "unknown")),
            "runtime_seconds": run_metadata.get("runtime_seconds"),
            "metrics": {
                "total_queries": total_queries,
                "classified_query_count": summary.get("classified_query_count", 0),
                "residual_query_count": summary.get("residual_query_count", 0),
                "residual_ratio": round(residual_ratio, 4),
                "total_categories": summary.get("total_categories", 0),
                "residual_cluster_count": summary.get("residual_cluster_count", 0),
                "long_tail_category_count_le_5": len(
                    [category for category in categories if category.get("query_count", 0) <= 5]
                ),
            },
            "top_categories": [
                {
                    "id": category["id"],
                    "name": category["name"],
                    "query_count": category["query_count"],
                    "definition": self._truncate(category.get("definition", ""), 240),
                    "boundary_notes": self._truncate(category.get("boundary_notes", ""), 220),
                    "positive_examples": category.get("positive_examples", [])[:3],
                }
                for category in categories[:12]
            ],
            "residual_clusters": [
                {
                    "id": cluster["id"],
                    "query_count": cluster["query_count"],
                    "reason": self._truncate(cluster.get("reason", ""), 200),
                    "sample_texts": cluster.get("sample_texts", [])[:3],
                }
                for cluster in residuals[:8]
            ],
            "hierarchy_outline": self._render_hierarchy_outline(
                (hierarchy or {}).get("roots", []),
                max_nodes=24,
            ),
        }

    def _build_llm_summary_prompt(
        self,
        payload: Dict[str, Any],
        *,
        language: str,
    ) -> tuple[str, str]:
        heading_map = {
            "zh": [
                "### 总体判断",
                "### 主要主题",
                "### 模糊与残差",
                "### 建议动作",
            ],
            "en": [
                "### Overall Takeaway",
                "### Important Themes",
                "### Residuals And Ambiguity",
                "### Recommended Actions",
            ],
        }
        headings = heading_map["zh" if language == "zh" else "en"]

        if language == "zh":
            system_prompt = (
                "你是 IntentAtlas 的分析报告撰写助手。"
                "你只能基于给定 JSON 写结论，不能编造不存在的类别、数量、原因或业务背景。"
                "输出必须是 Markdown，不要使用代码块。"
            )
            user_prompt = (
                "请把下面这份 taxonomy 运行结果写成适合人阅读的简洁分析摘要。\n"
                "要求：\n"
                f"1. 必须且仅能使用这 4 个标题：{', '.join(headings)}。\n"
                "2. 面向产品、运营或算法分析人员，语气克制，不写宣传话术。\n"
                "3. 优先引用具体主题名，尽量不要使用内部 ID，且不要把所有主题都复述一遍。\n"
                "4. 如果信息不足，请明确说明不确定，而不是脑补。\n"
                "5. 总长度控制在 250 到 500 字左右。\n\n"
                "输入 JSON：\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            )
            return system_prompt, user_prompt

        system_prompt = (
            "You write analyst-facing IntentAtlas summaries."
            " Use only the provided JSON and do not invent categories, counts, causes, or business context."
            " Return markdown only with no code fences."
        )
        user_prompt = (
            "Write a concise human-readable summary of this taxonomy run.\n"
            "Requirements:\n"
            f"1. Use exactly these four headings: {', '.join(headings)}.\n"
            "2. Write for product, operations, or ML analysts in a restrained factual tone.\n"
            "3. Prefer theme names over internal ids, and do not restate the whole taxonomy.\n"
            "4. If the data is insufficient, explicitly say so instead of guessing.\n"
            "5. Keep the response roughly 180 to 320 words.\n\n"
            "Input JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        return system_prompt, user_prompt

    def _render_hierarchy_outline(
        self,
        roots: list[Dict[str, Any]],
        *,
        max_nodes: int,
    ) -> list[str]:
        lines: list[str] = []

        def walk(node: Dict[str, Any], depth: int):
            if len(lines) >= max_nodes:
                return
            prefix = "  " * depth
            if "leaf_id" in node:
                lines.append(f"{prefix}- {node['leaf_id']}")
                return
            lines.append(
                f"{prefix}- {node.get('name', '').strip()}: {self._truncate(node.get('description', ''), 120)}"
            )
            for child in node.get("children", []):
                if len(lines) >= max_nodes:
                    break
                walk(child, depth + 1)

        for root in roots:
            if len(lines) >= max_nodes:
                break
            walk(root, 0)

        return lines

    def _sanitize_markdown_block(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()
        return stripped

    def _truncate(self, value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 3)].rstrip() + "..."

    def _llm_model_name(self, llm_service: Optional[Any]) -> Optional[str]:
        if llm_service is None:
            return None
        config = getattr(llm_service, "config", {}) or {}
        return config.get("model_name")
