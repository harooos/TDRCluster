"""
数据集管理器
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.config_loader import CONFIG
from .embedding_service import EmbeddingService
from core.state import Query


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self):
        self.config = CONFIG
        # 硬编码路径设置
        self.base_path = Path('data')
        self.raw_path = self.base_path / 'raw_data'
        self.processed_path = self.base_path / 'processed_data'
        
        # 确保目录存在
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_service = EmbeddingService()
    
    def _has_embeddings(self, dataset_name: str) -> bool:
        """检查数据集是否已有embeddings"""
        processed_dir = self.processed_path / dataset_name
        
        # 检查完整备份文件
        backup_file = processed_dir / f"{dataset_name}_complete.pkl"
        if backup_file.exists():
            return True
        
        # 检查分离文件
        metadata_file = processed_dir / f"{dataset_name}_metadata.csv"
        embeddings_file = processed_dir / f"{dataset_name}_embeddings.npz"
        
        return metadata_file.exists() and embeddings_file.exists()
    
    
    def generate_embeddings(self, dataset_name: str, force_regenerate: bool = False) -> str:
        """
        为数据集生成embeddings
        
        Args:
            dataset_name: 数据集名称
            force_regenerate: 是否强制重新生成
            
        Returns:
            str: 处理后数据的保存路径
        """
        # 检查是否已存在且不强制重新生成
        if not force_regenerate and self._has_embeddings(dataset_name):
            print(f"✓ 数据集 {dataset_name} 已有embeddings，直接复用")
            return str(self.processed_path / dataset_name)
        
        # 检查原始文件
        raw_file = self.raw_path / f"{dataset_name}.csv"
        if not raw_file.exists():
            raise FileNotFoundError(f"找不到原始数据文件: {raw_file}")
        
        print(f"🔄 开始为数据集 {dataset_name} 生成embeddings...")
        print(f"   原始文件: {raw_file}")
        
        # 读取原始数据
        try:
            df = pd.read_csv(raw_file)
            print(f"   读取到 {len(df)} 条记录")
        except Exception as e:
            raise ValueError(f"无法读取CSV文件: {e}")
        
        # 检查必要字段
        if 'query' not in df.columns:
            raise ValueError("CSV文件必须包含 'query' 列")
        
        # 准备数据
        embedding_service = self.embedding_service
        
        print("   开始生成embeddings...")
        
        # 批量处理 - doubao API限制每次最多256条
        batch_size = 256
        data_for_embedding = []
        
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            try:
                # 批量获取embeddings
                texts = [str(row['query']) for _, row in batch_df.iterrows()]
                embeddings = embedding_service.get_embeddings(texts)
                
                # 组装数据
                for (_, row), embedding in zip(batch_df.iterrows(), embeddings):
                    data_for_embedding.append({
                        'query': row['query'],
                        'category': row.get('category', ''),
                        'embedding': embedding
                    })
                
                print(f"   进度: {batch_end}/{len(df)} (批次处理)")
                    
            except Exception as e:
                print(f"   警告: 批次 {batch_start}-{batch_end} 失败，错误: {e}")
                continue
        
        # 验证生成的数据完整性
        if len(data_for_embedding) == 0:
            raise RuntimeError("未成功生成任何embeddings，不保存缓存文件")
        
        expected_count = len(df)
        actual_count = len(data_for_embedding)
        success_rate = actual_count / expected_count
        
        print(f"   Embedding生成统计:")
        print(f"     预期: {expected_count} 条")
        print(f"     实际: {actual_count} 条")
        print(f"     成功率: {success_rate:.2%}")
        
        # 如果成功率太低，不保存缓存
        if success_rate < 0.9:  # 成功率低于90%不保存
            raise RuntimeError(f"Embedding生成成功率过低 ({success_rate:.2%})，不保存缓存文件")
        
        # 保存数据
        saved_path = embedding_service.save_embeddings(data_for_embedding, dataset_name)
        print(f"✓ Embeddings已保存到: {saved_path}")
        
        return saved_path
    
    def load_dataset_as_queries(self, dataset_name: str, sample_size: Optional[int] = None) -> List[Query]:
        """
        加载数据集并转换为Query对象
        
        Args:
            dataset_name: 数据集名称
            sample_size: 采样大小（可选）
            
        Returns:
            List[Query]: Query对象列表
        """
        # 确保数据集有embeddings
        if not self._has_embeddings(dataset_name):
            print(f"⚠️ 数据集 {dataset_name} 没有embeddings，开始生成...")
            self.generate_embeddings(dataset_name)
        
        # 加载数据
        embedding_service = self.embedding_service
        data = embedding_service.load_embeddings(dataset_name)
        
        print(f"📂 成功加载数据集 {dataset_name}: {len(data)} 条记录")
        
        # 采样（如果需要）
        if sample_size and sample_size < len(data):
            import random
            data = random.sample(data, sample_size)
            print(f"   随机采样: {sample_size} 条记录")
        
        # 转换为Query对象
        queries = []
        for i, item in enumerate(data):
            query = Query(
                id=f"{dataset_name}-{i:04d}",
                content=item['query'],
                embedding=item['embedding']
            )
            queries.append(query)
        
        print(f"✓ 准备完成 {len(queries)} 个Query对象")
        return queries
    
