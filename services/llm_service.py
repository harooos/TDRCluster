"""
LLM服务
"""
from typing import Dict, Any, Optional, List
import xml.etree.ElementTree as ET
import re

from openai import OpenAI


class LLMService:
    """LLM服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                print("❌ LLM API Key未配置")
                return

            self.client = OpenAI(
                base_url=self.config.get("base_url"),
                api_key=api_key,
                timeout=60,
            )

            print("✓ LLM客户端初始化成功")
            print(f"   模型: {self.config.get('model_name', 'unknown')}")

        except Exception as e:
            print(f"❌ LLM客户端初始化失败: {e}")
            self.client = None

    def simple_call(self, prompt: str, system_message: Optional[str] = None) -> str:
        """LLM调用接口"""
        if not self.client:
            raise RuntimeError("LLM客户端未初始化")

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.get("model_name"),
                messages=messages,
                temperature=0.0,
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ LLM调用失败: {str(e)}")
            raise

    def analyze_cluster_with_retry(
        self,
        finalized_categories: List[Dict[str, Any]],
        cluster_to_review: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """评审单个micro-cluster，并返回唯一决策"""
        from core.prompts import create_review_prompt, create_reviewer_system_prompt

        prompt = create_review_prompt(finalized_categories, cluster_to_review)
        system_prompt = create_reviewer_system_prompt()
        existing_categories = {cat["id"]: cat for cat in finalized_categories}
        cluster_ids = [cluster_to_review["id"]]

        last_result = {"valid": False, "error_message": "unknown"}
        for attempt in range(max_retries + 1):
            if attempt == 0:
                print("🔍 调用LLM reviewer...")
            else:
                print(f"🔄 LLM reviewer重试 {attempt}/{max_retries} ...")

            try:
                response = self.simple_call(prompt, system_message=system_prompt)
                validation_result = self._validate_xml_response(
                    response,
                    cluster_ids,
                    existing_categories,
                )

                if validation_result["valid"]:
                    print("✅ LLM响应验证通过")
                    break

                print(f"❌ LLM响应验证失败: {validation_result['error_message']}")
                last_result = validation_result

            except Exception as e:
                print(f"❌ LLM reviewer调用失败: {str(e)}")
                last_result = {"valid": False, "error_message": str(e)}
        else:
            raise RuntimeError(f"LLM响应验证连续失败，最后错误: {last_result['error_message']}")

        decisions = validation_result.get("decisions", [])
        if len(decisions) != 1:
            raise RuntimeError(
                f"单簇评审期望1个决策，实际收到 {len(decisions)} 个"
            )

        return decisions[0]

    def _validate_xml_response(
        self,
        xml_response: str,
        cluster_ids: List[str],
        existing_categories: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """验证LLM返回的XML响应格式和完整性"""
        result = {
            "valid": False,
            "decisions": [],
            "error_message": "",
            "missing_clusters": [],
            "extra_clusters": [],
            "duplicate_clusters": [],
        }

        try:
            xml_response = xml_response.strip()
            if not xml_response.startswith("<"):
                start_pos = xml_response.find("<decisions>")
                if start_pos == -1:
                    result["error_message"] = "未找到<decisions>标签"
                    return result
                xml_response = xml_response[start_pos:]

            xml_response = re.sub(r"&(?!(?:amp|lt|gt|quot|apos);)", "&amp;", xml_response)
            root = ET.fromstring(xml_response)
            if root.tag != "decisions":
                result["error_message"] = f"根标签应为<decisions>，实际为<{root.tag}>"
                return result

            decisions = []
            decision_cluster_ids = set()

            for decision_elem in root.findall("decision"):
                id_elem = decision_elem.find("id")
                action_elem = decision_elem.find("action")

                if id_elem is None or not id_elem.text:
                    result["error_message"] = "决策缺少id字段"
                    return result
                if action_elem is None or not action_elem.text:
                    result["error_message"] = "决策缺少action字段"
                    return result

                cluster_id = id_elem.text.strip()
                if "," in cluster_id:
                    result["error_message"] = "每个决策只能引用一个micro-cluster id"
                    return result

                if cluster_id in decision_cluster_ids:
                    result["duplicate_clusters"].append(cluster_id)
                decision_cluster_ids.add(cluster_id)

                action = action_elem.text.strip().lower()

                decision = {
                    "id": cluster_id,
                    "action": action,
                }

                if action == "create":
                    name_elem = decision_elem.find("name")
                    definition_elem = decision_elem.find("definition")
                    boundary_elem = decision_elem.find("boundary_notes")

                    if name_elem is None or not name_elem.text:
                        result["error_message"] = "create决策缺少name字段"
                        return result
                    if definition_elem is None or not definition_elem.text:
                        result["error_message"] = "create决策缺少definition字段"
                        return result
                    if boundary_elem is None or not boundary_elem.text:
                        result["error_message"] = "create决策缺少boundary_notes字段"
                        return result

                    decision["name"] = name_elem.text.strip()
                    decision["definition"] = definition_elem.text.strip()
                    decision["boundary_notes"] = boundary_elem.text.strip()

                elif action == "merge":
                    target_elem = decision_elem.find("target_id")
                    name_update_elem = decision_elem.find("name_update")
                    definition_update_elem = decision_elem.find("definition_update")
                    boundary_update_elem = decision_elem.find("boundary_update")

                    if target_elem is None or not target_elem.text:
                        result["error_message"] = "merge决策缺少target_id字段"
                        return result
                    if name_update_elem is None or not name_update_elem.text:
                        result["error_message"] = "merge决策缺少name_update字段"
                        return result
                    if definition_update_elem is None or not definition_update_elem.text:
                        result["error_message"] = "merge决策缺少definition_update字段"
                        return result
                    if boundary_update_elem is None or not boundary_update_elem.text:
                        result["error_message"] = "merge决策缺少boundary_update字段"
                        return result

                    target_id = target_elem.text.strip()
                    if existing_categories is not None and target_id not in existing_categories:
                        result["error_message"] = f"merge决策的target_id '{target_id}' 不存在"
                        return result

                    decision["target_id"] = target_id
                    decision["name_update"] = name_update_elem.text.strip()
                    decision["definition_update"] = definition_update_elem.text.strip()
                    decision["boundary_update"] = boundary_update_elem.text.strip()

                elif action == "split":
                    reason_elem = decision_elem.find("reason")
                    if reason_elem is None or not reason_elem.text:
                        result["error_message"] = "split决策缺少reason字段"
                        return result
                    decision["reason"] = reason_elem.text.strip()
                else:
                    result["error_message"] = f"无效的action: {action}"
                    return result

                decisions.append(decision)

            expected_cluster_ids = set(cluster_ids)
            result["missing_clusters"] = list(expected_cluster_ids - decision_cluster_ids)
            result["extra_clusters"] = list(decision_cluster_ids - expected_cluster_ids)

            if result["missing_clusters"] or result["extra_clusters"] or result["duplicate_clusters"]:
                error_parts = []
                if result["missing_clusters"]:
                    error_parts.append(f"遗漏clusters: {result['missing_clusters']}")
                if result["extra_clusters"]:
                    error_parts.append(f"多余clusters: {result['extra_clusters']}")
                if result["duplicate_clusters"]:
                    error_parts.append(f"重复clusters: {result['duplicate_clusters']}")
                result["error_message"] = "; ".join(error_parts)
                return result

            result["valid"] = True
            result["decisions"] = decisions

        except ET.ParseError as e:
            result["error_message"] = f"XML解析失败: {str(e)}"
        except Exception as e:
            result["error_message"] = f"未知错误: {str(e)}"

        return result
