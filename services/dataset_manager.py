"""
数据集管理器
"""
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from config.config_loader import CONFIG
from .embedding_service import EmbeddingService
from core.state import Query
from services.run_control import CancelCallback, check_cancelled


class DatasetManager:
    """数据集管理器"""

    SUPPORTED_RAW_EXTENSIONS = (".csv", ".xlsx", ".xlsm", ".xls")

    def __init__(
        self,
        observer: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        should_cancel: CancelCallback = None,
    ):
        self.config = CONFIG
        # 硬编码路径设置
        self.base_path = Path("data")
        self.raw_path = self.base_path / "raw_data"
        self.processed_path = self.base_path / "processed_data"

        # 确保目录存在
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

        self.embedding_service = EmbeddingService()
        self.observer = observer
        self.should_cancel = should_cancel

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

    def _resolve_raw_file(self, dataset_name: str) -> Path:
        for extension in self.SUPPORTED_RAW_EXTENSIONS:
            candidate = self.raw_path / f"{dataset_name}{extension}"
            if candidate.exists():
                return candidate

        searched = ", ".join(f"{dataset_name}{extension}" for extension in self.SUPPORTED_RAW_EXTENSIONS)
        raise FileNotFoundError(
            f"找不到原始数据文件。已尝试: {searched}"
        )

    def _load_raw_dataframe(self, raw_file: Path) -> pd.DataFrame:
        suffix = raw_file.suffix.lower()

        try:
            if suffix == ".csv":
                return pd.read_csv(raw_file)
            if suffix in {".xlsx", ".xlsm", ".xls"}:
                return pd.read_excel(raw_file)
        except ImportError as exc:
            raise RuntimeError(
                "读取Excel文件失败，请确认已安装 openpyxl（.xlsx/.xlsm）或 xlrd（.xls）。"
            ) from exc
        except Exception as exc:
            raise ValueError(f"无法读取文件 {raw_file.name}: {exc}") from exc

        raise ValueError(f"不支持的文件格式: {raw_file.suffix}")

    def _normalize_dataframe(self, df: pd.DataFrame, raw_file: Path) -> pd.DataFrame:
        normalized_columns = {
            str(column).strip().lower(): column
            for column in df.columns
        }

        if "query" not in normalized_columns:
            raise ValueError(
                f"{raw_file.name} 必须包含 `query` 列（支持 CSV / Excel，列名大小写不敏感）"
            )

        renamed = df.rename(columns={normalized_columns["query"]: "query"}).copy()
        if "category" in normalized_columns:
            renamed = renamed.rename(columns={normalized_columns["category"]: "category"})
        elif "category" not in renamed.columns:
            renamed["category"] = ""

        renamed["query"] = renamed["query"].fillna("").astype(str).str.strip()
        renamed["category"] = renamed["category"].fillna("").astype(str).str.strip()
        renamed = renamed[renamed["query"] != ""].reset_index(drop=True)

        if renamed.empty:
            raise ValueError(f"{raw_file.name} 中没有可用的 query 数据")

        return renamed

    def _progress_file(self, dataset_name: str) -> Path:
        processed_dir = self.processed_path / dataset_name
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir / f"{dataset_name}_embedding_progress.pkl"

    def _load_embedding_progress(self, dataset_name: str) -> Dict[str, Any]:
        progress_file = self._progress_file(dataset_name)
        if not progress_file.exists():
            return {"next_index": 0, "data": []}

        with open(progress_file, "rb") as file:
            progress = pickle.load(file)

        next_index = int(progress.get("next_index", 0))
        data = progress.get("data", [])
        print(f"   检测到embedding断点缓存: 已完成 {next_index} 条")
        return {"next_index": next_index, "data": data}

    def _save_embedding_progress(self, dataset_name: str, next_index: int, data: List[Dict[str, Any]]):
        progress_file = self._progress_file(dataset_name)
        with open(progress_file, "wb") as file:
            pickle.dump({"next_index": next_index, "data": data}, file)

    def _clear_embedding_progress(self, dataset_name: str):
        progress_file = self._progress_file(dataset_name)
        if progress_file.exists():
            progress_file.unlink()

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
        raw_file = self._resolve_raw_file(dataset_name)

        print(f"🔄 开始为数据集 {dataset_name} 生成embeddings...")
        print(f"   原始文件: {raw_file}")

        # 读取原始数据
        df = self._normalize_dataframe(self._load_raw_dataframe(raw_file), raw_file)
        print(f"   读取到 {len(df)} 条记录")
        self._emit("embedding_started", dataset_name=dataset_name, total_rows=len(df))

        # 准备数据
        embedding_service = self.embedding_service

        print("   开始生成embeddings...")

        progress_file = self._progress_file(dataset_name)
        if force_regenerate and progress_file.exists():
            self._clear_embedding_progress(dataset_name)

        progress = self._load_embedding_progress(dataset_name)

        # 批量处理
        batch_size = 256
        data_for_embedding = progress["data"]
        start_index = progress["next_index"]

        for batch_start in range(start_index, len(df), batch_size):
            check_cancelled(self.should_cancel)
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            try:
                # 批量获取embeddings
                texts = [str(row["query"]) for _, row in batch_df.iterrows()]
                embeddings = embedding_service.get_embeddings(texts)

                # 组装数据
                for (_, row), embedding in zip(batch_df.iterrows(), embeddings):
                    data_for_embedding.append(
                        {
                            "query": row["query"],
                            "category": row.get("category", ""),
                            "embedding": embedding,
                        }
                    )

                self._save_embedding_progress(dataset_name, batch_end, data_for_embedding)
                print(f"   进度: {batch_end}/{len(df)} (批次处理)")
                self._emit(
                    "embedding_progress",
                    dataset_name=dataset_name,
                    completed_rows=batch_end,
                    total_rows=len(df),
                )

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
        self._clear_embedding_progress(dataset_name)
        print(f"✓ Embeddings已保存到: {saved_path}")
        self._emit(
            "embedding_completed",
            dataset_name=dataset_name,
            total_rows=len(data_for_embedding),
        )

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
        else:
            self._emit("embedding_reused", dataset_name=dataset_name)

        # 加载数据
        check_cancelled(self.should_cancel)
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
                content=item["query"],
                embedding=item["embedding"],
            )
            queries.append(query)

        print(f"✓ 准备完成 {len(queries)} 个Query对象")
        return queries

    def _emit(self, event: str, **payload):
        if self.observer:
            self.observer(event, payload)
