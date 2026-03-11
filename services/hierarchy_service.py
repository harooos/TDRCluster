"""
Post-hoc hierarchy induction service.
"""
from __future__ import annotations

import json
from typing import Dict, Any, List


class HierarchyService:
    """Builds a free-depth hierarchy over an existing flat taxonomy."""

    def __init__(self, llm_service):
        self.llm_service = llm_service

    def build_hierarchy_from_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        categories = self._extract_categories(summary)
        hierarchy = self.build_hierarchy(
            categories=categories,
            dataset_name=summary.get("dataset", "unknown"),
        )
        hierarchy["source_summary_timestamp"] = summary.get("timestamp")
        hierarchy["source_total_categories"] = len(categories)
        hierarchy["source_residual_query_count"] = summary.get("residual_query_count", 0)
        return hierarchy

    def build_hierarchy(
        self,
        categories: List[Dict[str, Any]],
        dataset_name: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        from core.hierarchy_prompts import (
            create_hierarchy_prompt,
            create_hierarchy_system_prompt,
            create_retry_suffix,
        )

        if not categories:
            raise ValueError("没有可用于构建层级的类别")

        base_prompt = create_hierarchy_prompt(categories, dataset_name)
        system_prompt = create_hierarchy_system_prompt()
        expected_leaf_ids = [category["id"] for category in categories]

        last_error = "unknown"
        for attempt in range(max_retries + 1):
            prompt = base_prompt
            if attempt > 0:
                prompt = f"{base_prompt}{create_retry_suffix(last_error)}"
                print(f"🔄 hierarchy builder重试 {attempt}/{max_retries} ...")
            else:
                print("🌲 调用LLM hierarchy builder...")

            try:
                response = self.llm_service.simple_call(
                    prompt,
                    system_message=system_prompt,
                    max_tokens=4000,
                    timeout=180,
                )
                validation = self._validate_hierarchy_response(response, expected_leaf_ids)
                if validation["valid"]:
                    print("✅ hierarchy JSON验证通过")
                    hierarchy = self._canonicalize_hierarchy(validation["hierarchy"])
                    hierarchy_stats = self._compute_hierarchy_stats(hierarchy["roots"])
                    hierarchy.update(
                        {
                            "dataset": dataset_name,
                            "leaf_category_count": len(expected_leaf_ids),
                            **hierarchy_stats,
                        }
                    )
                    return hierarchy

                last_error = validation["error_message"]
                print(f"❌ hierarchy JSON验证失败: {last_error}")
            except Exception as exc:
                last_error = str(exc)
                print(f"❌ hierarchy builder调用失败: {last_error}")

        raise RuntimeError(f"层级JSON连续验证失败，最后错误: {last_error}")

    def _extract_categories(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        categories = []
        for category_id, category in summary.get("categories", {}).items():
            categories.append(
                {
                    "id": category_id,
                    "name": category["name"],
                    "definition": category["definition"],
                    "boundary_notes": category.get("boundary_notes", ""),
                    "query_count": int(category.get("query_count", 0)),
                    "positive_examples": category.get("positive_examples", []),
                }
            )

        categories.sort(key=lambda item: (-item["query_count"], item["id"]))
        return categories

    def _validate_hierarchy_response(
        self,
        raw_response: str,
        expected_leaf_ids: List[str],
    ) -> Dict[str, Any]:
        result = {
            "valid": False,
            "hierarchy": None,
            "error_message": "",
            "stats": {},
        }

        try:
            hierarchy = self._extract_json_object(raw_response)
            if not isinstance(hierarchy, dict):
                result["error_message"] = "顶层必须是JSON对象"
                return result

            roots = hierarchy.get("roots")
            if not isinstance(roots, list) or not roots:
                result["error_message"] = "缺少非空 roots 数组"
                return result

            expected_leaf_id_set = set(expected_leaf_ids)
            seen_leaf_ids: List[str] = []

            def walk(node: Dict[str, Any], depth: int, path: str):
                if not isinstance(node, dict):
                    raise ValueError(f"{path} 必须是对象")

                is_leaf = "leaf_id" in node
                is_internal = {"name", "description", "children"}.issubset(node.keys())

                if is_leaf == is_internal:
                    raise ValueError(
                        f"{path} 必须是 leaf 节点或 internal 节点之一，且不能同时属于两者"
                    )

                if is_leaf:
                    if set(node.keys()) != {"leaf_id"}:
                        raise ValueError(f"{path} leaf 节点只能包含 leaf_id 字段")
                    leaf_id = node["leaf_id"]
                    if not isinstance(leaf_id, str) or not leaf_id.strip():
                        raise ValueError(f"{path} leaf_id 必须是非空字符串")
                    seen_leaf_ids.append(leaf_id)
                    return

                if set(node.keys()) != {"name", "description", "children"}:
                    raise ValueError(
                        f"{path} internal 节点只能包含 name / description / children 字段"
                    )

                name = node["name"]
                description = node["description"]
                children = node["children"]

                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"{path} internal 节点 name 必须是非空字符串")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError(f"{path} internal 节点 description 必须是非空字符串")
                if not isinstance(children, list) or not children:
                    raise ValueError(f"{path} internal 节点 children 必须是非空数组")

                for child_index, child in enumerate(children, 1):
                    walk(child, depth + 1, f"{path}.children[{child_index}]")

            for index, root in enumerate(roots, 1):
                walk(root, depth=1, path=f"roots[{index}]")

            seen_leaf_id_set = set(seen_leaf_ids)
            missing_leaf_ids = sorted(expected_leaf_id_set - seen_leaf_id_set)
            extra_leaf_ids = sorted(seen_leaf_id_set - expected_leaf_id_set)
            duplicate_leaf_ids = sorted(
                {
                    leaf_id
                    for leaf_id in seen_leaf_ids
                    if seen_leaf_ids.count(leaf_id) > 1
                }
            )

            if missing_leaf_ids or extra_leaf_ids or duplicate_leaf_ids:
                issues = []
                if missing_leaf_ids:
                    issues.append(f"缺少leaf_ids: {missing_leaf_ids}")
                if extra_leaf_ids:
                    issues.append(f"未知leaf_ids: {extra_leaf_ids}")
                if duplicate_leaf_ids:
                    issues.append(f"重复leaf_ids: {duplicate_leaf_ids}")
                result["error_message"] = "; ".join(issues)
                return result

            result["valid"] = True
            result["hierarchy"] = {
                "roots": roots,
            }
            result["stats"] = {}
            return result

        except Exception as exc:
            result["error_message"] = str(exc)
            return result

    def _extract_json_object(self, raw_response: str) -> Dict[str, Any]:
        raw_response = raw_response.strip()
        if raw_response.startswith("```"):
            lines = raw_response.splitlines()
            if len(lines) >= 3:
                raw_response = "\n".join(lines[1:-1]).strip()

        start_index = raw_response.find("{")
        end_index = raw_response.rfind("}")
        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise ValueError("未找到合法JSON对象")

        json_text = raw_response[start_index:end_index + 1]
        return json.loads(json_text)

    def _canonicalize_hierarchy(self, hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        canonical_roots = []
        for root in hierarchy["roots"]:
            canonical_root = self._canonicalize_node(root)
            if isinstance(canonical_root, list):
                canonical_roots.extend(canonical_root)
            else:
                canonical_roots.append(canonical_root)
        return {"roots": canonical_roots}

    def _canonicalize_node(self, node: Dict[str, Any]):
        if "leaf_id" in node:
            return node

        canonical_children = []
        for child in node["children"]:
            canonical_child = self._canonicalize_node(child)
            if isinstance(canonical_child, list):
                canonical_children.extend(canonical_child)
            else:
                canonical_children.append(canonical_child)

        node = {
            "name": node["name"].strip(),
            "description": node["description"].strip(),
            "children": canonical_children,
        }

        if len(canonical_children) == 1:
            return canonical_children[0]

        return node

    def _compute_hierarchy_stats(self, roots: List[Dict[str, Any]]) -> Dict[str, Any]:
        internal_node_count = 0
        max_depth = 0
        single_child_internal_count = 0

        def walk(node: Dict[str, Any], depth: int):
            nonlocal internal_node_count, max_depth, single_child_internal_count
            max_depth = max(max_depth, depth)

            if "leaf_id" in node:
                return

            internal_node_count += 1
            if len(node["children"]) == 1:
                single_child_internal_count += 1

            for child in node["children"]:
                walk(child, depth + 1)

        for root in roots:
            walk(root, 1)

        return {
            "root_count": len(roots),
            "internal_node_count": internal_node_count,
            "max_depth": max_depth,
            "single_child_internal_count": single_child_internal_count,
        }
