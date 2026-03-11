"""
Embedding服务
"""
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from urllib import request, error

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from openai import OpenAI

from config.config_loader import CONFIG


class EmbeddingService:
    """Embedding服务"""
    
    def __init__(self):
        self.config = CONFIG.get('embedding', {})
        self.api_style = self.config.get('api_style', 'openai')
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化embedding客户端"""
        try:
            api_key = self.config.get('api_key')
            
            if not api_key:
                print("❌ Embedding API Key未配置")
                return

            if self.api_style == 'openai':
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=self.config.get('base_url'),
                    timeout=60
                )
            elif self.api_style == 'ark_multimodal':
                self.client = True
            else:
                raise ValueError(f"不支持的embedding api_style: {self.api_style}")

            print(f"✓ Embedding客户端初始化成功")
            print(f"   接口风格: {self.api_style}")
            print(f"   模型: {self.config.get('model_name', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Embedding客户端初始化失败: {e}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的embedding向量"""
        if not self.client:
            raise RuntimeError("Embedding客户端未初始化")
        
        if not texts:
            return []
            
        try:
            cleaned_texts = [text.replace("\n", " ") for text in texts]
            if self.api_style == 'openai':
                response = self.client.embeddings.create(
                    input=cleaned_texts,
                    model=self.config.get('model_name')
                )
                return [item.embedding for item in response.data]

            if self.api_style == 'ark_multimodal':
                return self._get_ark_multimodal_embeddings(cleaned_texts)

            raise ValueError(f"不支持的embedding api_style: {self.api_style}")
        except Exception as e:
            print(f"❌ 获取embeddings失败: {str(e)}")
            raise

    def _get_ark_multimodal_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        使用方舟 /embeddings/multimodal 接口生成文本embedding。
        该接口一次请求返回一个embedding，因此这里对输入文本逐条请求，
        再通过线程池做轻量并发以减少总耗时。
        """
        max_concurrency = max(1, int(self.config.get('max_concurrency', 16)))
        if len(texts) == 1:
            return [self._request_ark_multimodal_embedding(texts[0])]

        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(texts))) as executor:
            return list(executor.map(self._request_ark_multimodal_embedding, texts))

    def _request_ark_multimodal_embedding(self, text: str) -> List[float]:
        endpoint_path = self.config.get('endpoint_path', '/embeddings/multimodal')
        base_url = self.config.get('base_url', '').rstrip('/')
        url = f"{base_url}{endpoint_path}"

        payload = {
            "model": self.config.get('model_name'),
            "input": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
        }

        raw_body = json.dumps(payload).encode('utf-8')
        http_request = request.Request(
            url=url,
            data=raw_body,
            method='POST',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {self.config.get('api_key')}",
                'Connection': 'close',
            },
        )

        timeout_seconds = int(self.config.get('timeout', 60))
        max_retries = max(1, int(self.config.get('max_retries', 4)))
        retry_delay_seconds = float(self.config.get('retry_delay_seconds', 1.0))

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                with request.urlopen(http_request, timeout=timeout_seconds) as response:
                    response_body = response.read().decode('utf-8')
                response_json = json.loads(response_body)
                embedding = response_json.get('data', {}).get('embedding')
                if not embedding:
                    raise RuntimeError(f"方舟multimodal embedding响应缺少embedding字段: {response_json}")
                return embedding
            except error.HTTPError as http_error:
                error_body = http_error.read().decode('utf-8', errors='replace')
                last_error = RuntimeError(
                    f"方舟multimodal embedding请求失败: {http_error.code} {error_body}"
                )
            except Exception as exc:
                last_error = exc

            if attempt < max_retries:
                time.sleep(retry_delay_seconds * attempt)

        raise RuntimeError(f"方舟multimodal embedding重试失败: {last_error}") from last_error
    
    def save_embeddings(self, data: List[Dict[str, Any]], dataset_name: str) -> str:
        """保存embeddings数据"""
        base_path = Path(f"data/processed_data/{dataset_name}")
        base_path.mkdir(parents=True, exist_ok=True)
        
        queries = [item['query'] for item in data]
        categories = [item.get('category', '') for item in data]
        embeddings = np.array([item['embedding'] for item in data])
        
        # 保存元数据
        metadata_df = pd.DataFrame({
            'id': range(len(queries)),
            'query': queries,
            'category': categories
        })
        metadata_path = base_path / f"{dataset_name}_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        # 保存向量数据
        embeddings_path = base_path / f"{dataset_name}_embeddings.npz"
        np.savez_compressed(embeddings_path, embeddings=embeddings)
        
        # 保存完整备份
        backup_path = base_path / f"{dataset_name}_complete.pkl"
        with open(backup_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Embeddings已保存 ({len(data)} 条记录)")
        return str(base_path)
    
    def load_embeddings(self, dataset_name: str) -> List[Dict[str, Any]]:
        """加载embeddings数据"""
        base_path = Path(f"data/processed_data/{dataset_name}")
        
        # 优先加载完整备份
        backup_path = base_path / f"{dataset_name}_complete.pkl"
        if backup_path.exists():
            with open(backup_path, 'rb') as f:
                return pickle.load(f)
        
        # 从分离文件加载
        metadata_path = base_path / f"{dataset_name}_metadata.csv"
        embeddings_path = base_path / f"{dataset_name}_embeddings.npz"
        
        if not (metadata_path.exists() and embeddings_path.exists()):
            raise FileNotFoundError(f"找不到数据集 {dataset_name} 的embeddings文件")
        
        metadata_df = pd.read_csv(metadata_path)
        embeddings_data = np.load(embeddings_path)
        embeddings = embeddings_data['embeddings']
        
        result = []
        for i, (_, row) in enumerate(metadata_df.iterrows()):
            result.append({
                'query': row['query'],
                'category': row['category'],
                'embedding': embeddings[i].tolist()
            })
        
        return result
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        try:
            data = self.load_embeddings(dataset_name)
            categories = [item.get('category', '') for item in data]
            unique_categories = set(categories)
            
            return {
                'total_queries': len(data),
                'unique_categories': len(unique_categories),
                'categories': sorted(list(unique_categories)),
                'embedding_dim': len(data[0]['embedding']) if data else 0,
                'sample_queries': [item['query'] for item in data[:3]]
            }
        except Exception as e:
            return {'error': str(e)}
