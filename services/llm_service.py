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
            api_key = self.config.get('api_key')
            if not api_key:
                print("❌ LLM API Key未配置")
                return
            
            self.client = OpenAI(
                base_url=self.config.get('base_url'),
                api_key=api_key,
                timeout=60  # 硬编码1分钟超时
            )
            
            print(f"✓ LLM客户端初始化成功")
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
                model=self.config.get('model_name'),
                messages=messages,
                temperature=0.0,  # 硬编码确定性输出
                max_tokens=2000   # 硬编码足够的token限制
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ LLM调用失败: {str(e)}")
            raise
    
    def analyze_clusters_with_retry(self, finalized_categories: List[Dict[str, Any]], 
                                   clusters_to_review: List[Dict[str, Any]],
                                   dataset_name: str = None,
                                   max_retries: int = 3) -> Dict[str, Any]:
        """
        高级cluster分析方法：包含prompt生成、调用、验证、重试的完整流程
        
        Args:
            finalized_categories: 已确定的类别列表
            clusters_to_review: 待评审的clusters列表
            dataset_name: 数据集名称，用于获取特定配置
            max_retries: 最大重试次数
            
        Returns:
            Dict[str, Any]: 验证通过的解析结果
        """
        # 导入prompt生成函数
        from core.prompts import create_review_prompt
        
        # 生成prompt
        prompt = create_review_prompt(finalized_categories, clusters_to_review, dataset_name)
        
        # 构建existing_categories映射（用于验证）
        existing_categories = {cat['id']: cat for cat in finalized_categories}
        cluster_ids = [cluster['id'] for cluster in clusters_to_review]
        
        # 首次尝试
        try:
            response = self.simple_call(prompt)
            validation_result = self._validate_xml_response(response, cluster_ids, existing_categories)
            
            if validation_result['valid']:
                print("✅ LLM响应验证通过")
                return validation_result
            else:
                print(f"❌ LLM响应验证失败: {validation_result['error_message']}")
                
        except Exception as e:
            print(f"❌ 初次LLM调用失败: {str(e)}")
            validation_result = {'valid': False, 'error_message': f'LLM调用异常: {str(e)}'}
        
        # 重试机制
        print("🔄 开始重试机制...")
        for attempt in range(max_retries):
            print(f"   尝试 {attempt + 1}/{max_retries}: 重新调用LLM")
            
            try:
                response = self.simple_call(prompt)
                validation_result = self._validate_xml_response(response, cluster_ids, existing_categories)
                
                if validation_result['valid']:
                    print(f"   ✅ 重试成功，验证通过")
                    return validation_result
                else:
                    print(f"   ❌ 重试 {attempt + 1} 验证失败: {validation_result['error_message']}")
                    
            except Exception as e:
                print(f"   ❌ 重试 {attempt + 1} 调用失败: {str(e)}")
        
        # 所有重试都失败
        print("❌ 所有重试失败，LLM无法正确响应")
        raise RuntimeError(f"LLM响应验证连续失败，最后错误: {validation_result['error_message']}")
    
    def _validate_xml_response(self, xml_response: str, cluster_ids: List[str], 
                              existing_categories: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        验证LLM返回的XML响应格式和完整性
        
        Args:
            xml_response: LLM返回的XML字符串
            cluster_ids: 应该被判断的cluster ID列表
            existing_categories: 现有类别映射，用于验证assign操作
            
        Returns:
            Dict[str, Any]: 验证结果，包含是否有效、解析后的决策列表、错误信息
        """
        result = {
            'valid': False,
            'decisions': [],
            'error_message': '',
            'missing_clusters': [],
            'extra_clusters': [],
            'duplicate_clusters': []
        }
        
        try:
            # 清理和预处理XML
            xml_response = xml_response.strip()
            if not xml_response.startswith('<'):
                # 寻找XML开始标记
                start_pos = xml_response.find('<decisions>')
                if start_pos == -1:
                    result['error_message'] = "未找到<decisions>标签"
                    return result
                xml_response = xml_response[start_pos:]
            
            # 自动修复常见的XML格式问题
            # 只转义那些不是XML实体的&符号
            xml_response = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', xml_response)
            
            # 解析XML
            root = ET.fromstring(xml_response)
            
            if root.tag != 'decisions':
                result['error_message'] = f"根标签应为<decisions>，实际为<{root.tag}>"
                return result
            
            # 解析所有决策
            decisions = []
            decision_cluster_ids = set()
            
            for decision_elem in root.findall('decision'):
                decision = {}
                
                # 获取ID
                id_elem = decision_elem.find('id')
                if id_elem is None or not id_elem.text:
                    result['error_message'] = "决策缺少id字段"
                    return result
                
                decision['id'] = id_elem.text.strip()
                
                # 获取action
                action_elem = decision_elem.find('action')
                if action_elem is None or not action_elem.text:
                    result['error_message'] = "决策缺少action字段"
                    return result
                
                action = action_elem.text.strip().lower()
                decision['action'] = action
                
                # 根据action获取额外参数
                if action == 'create':
                    desc_elem = decision_elem.find('description')
                    if desc_elem is None or not desc_elem.text:
                        result['error_message'] = "create决策缺少description字段"
                        return result
                    decision['description'] = desc_elem.text.strip()
                    
                elif action == 'assign':
                    target_elem = decision_elem.find('target_id')
                    if target_elem is None or not target_elem.text:
                        result['error_message'] = "assign决策缺少target_id字段"
                        return result
                    target_id = target_elem.text.strip()
                    decision['target_id'] = target_id
                    
                    # 校验target_id是否存在（如果提供了existing_categories）
                    if existing_categories is not None and target_id not in existing_categories:
                        result['error_message'] = f"assign决策的target_id '{target_id}' 不存在于现有categories中"
                        return result
                    
                    # 检查description_update字段
                    desc_update_elem = decision_elem.find('description_update')
                    if desc_update_elem is None or not desc_update_elem.text:
                        result['error_message'] = "assign决策缺少description_update字段"
                        return result
                    decision['description_update'] = desc_update_elem.text.strip()
                    
                elif action == 'subdivide':
                    k_elem = decision_elem.find('k_value')
                    if k_elem is None or not k_elem.text:
                        result['error_message'] = "subdivide决策缺少k_value字段"
                        return result
                    try:
                        decision['k_value'] = int(k_elem.text.strip())
                    except ValueError:
                        result['error_message'] = f"k_value必须为整数: {k_elem.text}"
                        return result
                else:
                    result['error_message'] = f"无效的action: {action}"
                    return result
                
                # 收集决策涉及的cluster IDs
                cluster_ids_in_decision = [cid.strip() for cid in decision['id'].split(',')]
                for cid in cluster_ids_in_decision:
                    if cid in decision_cluster_ids:
                        result['duplicate_clusters'].append(cid)
                    decision_cluster_ids.add(cid)
                
                decisions.append(decision)
            
            # 检查完整性
            expected_cluster_ids = set(cluster_ids)
            result['missing_clusters'] = list(expected_cluster_ids - decision_cluster_ids)
            result['extra_clusters'] = list(decision_cluster_ids - expected_cluster_ids)
            
            # 如果有重复、遗漏或多余的clusters，标记为无效
            if result['missing_clusters'] or result['extra_clusters'] or result['duplicate_clusters']:
                error_parts = []
                if result['missing_clusters']:
                    error_parts.append(f"遗漏clusters: {result['missing_clusters']}")
                if result['extra_clusters']:
                    error_parts.append(f"多余clusters: {result['extra_clusters']}")
                if result['duplicate_clusters']:
                    error_parts.append(f"重复clusters: {result['duplicate_clusters']}")
                result['error_message'] = "; ".join(error_parts)
                return result
            
            # 验证通过
            result['valid'] = True
            result['decisions'] = decisions
            
        except ET.ParseError as e:
            result['error_message'] = f"XML解析失败: {str(e)}"
        except Exception as e:
            result['error_message'] = f"未知错误: {str(e)}"
        
        return result
    