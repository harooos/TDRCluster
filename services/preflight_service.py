"""
Dataset preflight checks and rough cost estimation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config.config_loader import CONFIG
from services.clustering_service import ClusteringService


@dataclass
class PricingConfig:
    embedding_input_per_million_tokens: float = 0.0
    llm_input_per_million_tokens: float = 0.0
    llm_output_per_million_tokens: float = 0.0


DEFAULT_MODEL_PRICING = {
    "embedding": {
        # Based on the user's current preferred Doubao embedding pricing input.
        "doubao-embedding-vision-251215": 0.7,
        "doubao-embedding-large": 0.7,
    },
    "llm": {
        # Default to the small-context tier for doubao-seed-2.0-mini, which matches this project's typical prompt size.
        "doubao-seed-2-0-mini-260215": {
            "input_per_million_tokens": 0.2,
            "output_per_million_tokens": 2.0,
        },
        "doubao-seed-2.0-mini": {
            "input_per_million_tokens": 0.2,
            "output_per_million_tokens": 2.0,
        },
    },
}


def resolve_pricing_config(config: Dict[str, Any]) -> PricingConfig:
    pricing = config.get("pricing", {})
    llm_model_name = (config.get("llm", {}) or {}).get("model_name", "")
    embedding_model_name = (config.get("embedding", {}) or {}).get("model_name", "")
    llm_defaults = DEFAULT_MODEL_PRICING["llm"].get(llm_model_name, {})
    embedding_default = DEFAULT_MODEL_PRICING["embedding"].get(embedding_model_name, 0.0)

    return PricingConfig(
        embedding_input_per_million_tokens=_resolve_price_value(
            primary=pricing.get("embedding_input_per_million_tokens"),
            legacy=pricing.get("embedding_input_per_1k_tokens"),
            default=embedding_default,
        ),
        llm_input_per_million_tokens=_resolve_price_value(
            primary=pricing.get("llm_input_per_million_tokens"),
            legacy=pricing.get("llm_input_per_1k_tokens"),
            default=llm_defaults.get("input_per_million_tokens", 0.0),
        ),
        llm_output_per_million_tokens=_resolve_price_value(
            primary=pricing.get("llm_output_per_million_tokens"),
            legacy=pricing.get("llm_output_per_1k_tokens"),
            default=llm_defaults.get("output_per_million_tokens", 0.0),
        ),
    )


def _resolve_price_value(
    *,
    primary: Any,
    legacy: Any,
    default: float,
) -> float:
    primary_value = _positive_float_or_none(primary)
    if primary_value is not None:
        return primary_value

    legacy_value = _positive_float_or_none(legacy)
    if legacy_value is not None:
        return legacy_value * 1000.0

    return float(default)


def _positive_float_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


class PreflightService:
    """Lightweight checks before an expensive run."""

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xlsm", ".xls"}

    def load_dataframe_from_upload(self, filename: str, content: bytes) -> pd.DataFrame:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        buffer = BytesIO(content)
        if suffix == ".csv":
            return pd.read_csv(buffer)
        return pd.read_excel(buffer)

    def normalize_dataframe(self, df: pd.DataFrame, *, query_col: str) -> pd.DataFrame:
        if query_col not in df.columns:
            raise ValueError(f"Missing query column: {query_col}")

        normalized = df.copy()
        normalized = normalized.rename(columns={query_col: "query"})
        normalized["query"] = normalized["query"].fillna("").astype(str).str.strip()
        normalized["_is_empty"] = normalized["query"] == ""
        return normalized

    def analyze(
        self,
        df: pd.DataFrame,
        *,
        query_col: str,
        pricing: Optional[PricingConfig] = None,
    ) -> Dict[str, Any]:
        normalized = self.normalize_dataframe(df, query_col=query_col)
        valid_queries = normalized.loc[~normalized["_is_empty"], "query"]
        total_rows = int(len(normalized))
        valid_rows = int(len(valid_queries))
        duplicate_rows = int(valid_queries.duplicated().sum())
        empty_rows = int(normalized["_is_empty"].sum())
        avg_chars = float(valid_queries.str.len().mean()) if valid_rows else 0.0
        p95_chars = float(valid_queries.str.len().quantile(0.95)) if valid_rows else 0.0
        total_text_tokens = int(sum(self.estimate_tokens(text) for text in valid_queries.tolist()))
        suggested_preview_rows = self.recommend_preview_rows(valid_rows)

        clustering_service = ClusteringService()
        min_cluster_size = clustering_service.calculate_min_cluster_size(max(valid_rows, 1))
        pricing = pricing or PricingConfig()
        estimated_microclusters = self.estimate_microcluster_count(valid_rows, min_cluster_size)
        avg_tokens_per_row = (total_text_tokens / valid_rows) if valid_rows else 0.0
        preview_text_tokens = int(round(avg_tokens_per_row * suggested_preview_rows))

        full_run_estimate = self.build_run_cost_estimate(
            row_count=valid_rows,
            text_tokens=total_text_tokens,
            min_cluster_size=min_cluster_size,
            pricing=pricing,
        )
        preview_run_estimate = self.build_run_cost_estimate(
            row_count=suggested_preview_rows,
            text_tokens=preview_text_tokens,
            min_cluster_size=clustering_service.calculate_min_cluster_size(max(suggested_preview_rows, 1)),
            pricing=pricing,
        )

        return {
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "empty_rows": empty_rows,
            "duplicate_rows": duplicate_rows,
            "duplicate_ratio": duplicate_rows / max(valid_rows, 1),
            "avg_chars": avg_chars,
            "p95_chars": p95_chars,
            "estimated_embedding_tokens": total_text_tokens,
            "estimated_microclusters": estimated_microclusters,
            "min_cluster_size": min_cluster_size,
            "suggested_preview_rows": suggested_preview_rows,
            "cost_estimate": {
                "preview_run": preview_run_estimate,
                "full_run": full_run_estimate,
            },
            "quality_flags": self.build_quality_flags(
                total_rows=total_rows,
                valid_rows=valid_rows,
                empty_rows=empty_rows,
                duplicate_rows=duplicate_rows,
                p95_chars=p95_chars,
            ),
            "columns": list(df.columns),
        }

    def build_run_cost_estimate(
        self,
        *,
        row_count: int,
        text_tokens: int,
        min_cluster_size: int,
        pricing: PricingConfig,
    ) -> Dict[str, Any]:
        estimated_microclusters = self.estimate_microcluster_count(row_count, min_cluster_size)
        llm_estimate = self.estimate_llm_tokens(
            total_rows=row_count,
            total_text_tokens=text_tokens,
            estimated_microclusters=estimated_microclusters,
        )

        llm_input_mid = (llm_estimate["input_tokens_low"] + llm_estimate["input_tokens_high"]) / 2.0
        llm_output_mid = (llm_estimate["output_tokens_low"] + llm_estimate["output_tokens_high"]) / 2.0
        llm_total_mid = llm_input_mid + llm_output_mid

        embedding_cost = (
            round(text_tokens / 1_000_000.0 * pricing.embedding_input_per_million_tokens, 6)
            if pricing.embedding_input_per_million_tokens > 0
            else None
        )
        llm_cost = None
        if pricing.llm_input_per_million_tokens > 0 or pricing.llm_output_per_million_tokens > 0:
            llm_cost = round(
                llm_input_mid / 1_000_000.0 * pricing.llm_input_per_million_tokens
                + llm_output_mid / 1_000_000.0 * pricing.llm_output_per_million_tokens,
                6,
            )

        total_cost = None
        if embedding_cost is not None or llm_cost is not None:
            total_cost = round((embedding_cost or 0.0) + (llm_cost or 0.0), 6)

        return {
            "row_count": row_count,
            "embedding_tokens": text_tokens,
            "estimated_microclusters": estimated_microclusters,
            "llm_input_tokens": int(round(llm_input_mid)),
            "llm_output_tokens": int(round(llm_output_mid)),
            "llm_total_tokens": int(round(llm_total_mid)),
            "embedding_cost_rmb": embedding_cost,
            "llm_cost_rmb": llm_cost,
            "total_cost_rmb": total_cost,
        }

    def estimate_tokens(self, text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0

        latin_like = stripped.encode("utf-8", errors="ignore").decode("utf-8")
        whitespace_units = len(latin_like.split())
        char_units = math.ceil(len(stripped) / 4)
        return max(1, whitespace_units, char_units)

    def recommend_preview_rows(self, valid_rows: int) -> int:
        if valid_rows <= 0:
            return 0
        return min(valid_rows, 512)

    def estimate_microcluster_count(self, valid_rows: int, min_cluster_size: int) -> int:
        if valid_rows <= 0:
            return 0
        heuristic = round(valid_rows / max(min_cluster_size * 2.2, 24))
        return max(8, min(200, heuristic))

    def estimate_llm_tokens(
        self,
        *,
        total_rows: int,
        total_text_tokens: int,
        estimated_microclusters: int,
    ) -> Dict[str, int]:
        clustering_config = CONFIG.get("clustering", {})
        sample_budget = int(clustering_config.get("max_samples_per_cluster", 16))
        avg_query_tokens = total_text_tokens / max(total_rows, 1)
        evidence_tokens = sample_budget * avg_query_tokens
        taxonomy_growth_factor = math.log2(max(estimated_microclusters, 2))

        input_tokens_low = int(
            estimated_microclusters * (900 + evidence_tokens * 1.1 + taxonomy_growth_factor * 140)
        )
        input_tokens_high = int(
            estimated_microclusters * (1400 + evidence_tokens * 1.7 + taxonomy_growth_factor * 220)
        )
        output_tokens_low = int(estimated_microclusters * 160)
        output_tokens_high = int(estimated_microclusters * 420)

        return {
            "input_tokens_low": input_tokens_low,
            "input_tokens_high": input_tokens_high,
            "output_tokens_low": output_tokens_low,
            "output_tokens_high": output_tokens_high,
        }

    def build_quality_flags(
        self,
        *,
        total_rows: int,
        valid_rows: int,
        empty_rows: int,
        duplicate_rows: int,
        p95_chars: float,
    ) -> list[str]:
        flags = []
        if valid_rows == 0:
            flags.append("No valid rows found in the selected query column.")
            return flags
        if empty_rows / max(total_rows, 1) > 0.05:
            flags.append("More than 5% of rows are empty after trimming.")
        if duplicate_rows / max(valid_rows, 1) > 0.2:
            flags.append("Duplicate query ratio is above 20%; deduplication may improve signal quality.")
        if p95_chars > 250:
            flags.append("The long-tail query length is high; consider isolating verbose records or transcripts.")
        if valid_rows < 100:
            flags.append("Dataset is small; preview and full run will be close in quality.")
        return flags
