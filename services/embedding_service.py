"""
Embedding服务
"""
import pickle
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
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            api_key = self.config.get('api_key')
            
            if not api_key:
                print("❌ Embedding API Key未配置")
                return
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.config.get('base_url'),
                timeout=60  # 硬编码1分钟超时
            )
            
            print(f"✓ Embedding客户端初始化成功")
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
            response = self.client.embeddings.create(
                input=cleaned_texts,
                model=self.config.get('model_name')
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"❌ 获取embeddings失败: {str(e)}")
            raise
    
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