"""
Persisted project manifest helpers for history, recovery, and reload.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config.config_loader import CONFIG
from services.preflight_service import PreflightService, resolve_pricing_config
from services.report_service import ReportService


@dataclass
class RunRecord:
    run_id: str
    dataset_name: str
    display_name: str
    timestamp: str
    run_mode: str
    status: str
    source: str
    summary_path: Optional[str] = None
    hierarchy_path: Optional[str] = None
    csv_path: Optional[str] = None
    agent_report_path: Optional[str] = None
    runtime_seconds: Optional[float] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    report_language: Optional[str] = None
    source_data_path: Optional[str] = None
    source_row_count: Optional[int] = None
    uploaded_name: Optional[str] = None
    query_column: Optional[str] = None
    preview_rows: Optional[int] = None
    build_hierarchy: Optional[bool] = None
    preflight: Optional[Dict[str, Any]] = None
    session_snapshot: Optional[Dict[str, Any]] = None
    resume_state_path: Optional[str] = None
    manifest_path: Optional[str] = None


class RunHistoryService:
    """Indexes saved runs and loads persisted project context."""

    def __init__(self, output_dir: str | Path = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_service = ReportService()
        self.source_dir = self.output_dir / "project_sources"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = Path("data/raw_data")

    def manifest_path(self, run_id: str) -> Path:
        return self.output_dir / f"run_manifest_{run_id}.json"

    def save_source_dataframe(self, run_id: str, df: pd.DataFrame) -> Path:
        path = self.source_dir / f"{run_id}_source.csv"
        df.to_csv(path, index=False)
        return path

    def load_source_dataframe(self, source_path: str | Path) -> pd.DataFrame:
        return pd.read_csv(self._resolve_path(source_path))

    def upsert_manifest(self, record: Dict[str, Any]) -> Path:
        run_id = record["run_id"]
        manifest_path_value = record.get("manifest_path")
        path = self._resolve_path(manifest_path_value) if manifest_path_value else self.manifest_path(run_id)
        payload = record.copy()
        payload["manifest_path"] = str(path)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_manifest(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self.manifest_path(run_id)
        if not path.exists():
            for manifest_path in self.output_dir.glob("run_manifest_*.json"):
                try:
                    record = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if record.get("run_id") == run_id:
                    record["manifest_path"] = str(manifest_path)
                    return record
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        record["manifest_path"] = str(path)
        return record

    def list_runs(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        seen_summary_paths: set[str] = set()

        for manifest_path in sorted(self.output_dir.glob("run_manifest_*.json"), reverse=True):
            try:
                record = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            record["manifest_path"] = str(manifest_path)
            records.append(record)
            summary_path = record.get("summary_path")
            if summary_path:
                seen_summary_paths.add(str(self._resolve_path(summary_path)))

        for record in self._load_legacy_records():
            summary_path = record.get("summary_path")
            if summary_path and str(self._resolve_path(summary_path)) in seen_summary_paths:
                continue
            records.append(record)

        records.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return records

    def load_project_bundle(
        self,
        record: Dict[str, Any],
        *,
        report_language: str = "zh",
    ) -> Dict[str, Any]:
        record = record.copy()
        source_df = None
        preflight = record.get("preflight")
        source_data_path = record.get("source_data_path")
        query_column = record.get("query_column") or "query"

        if source_data_path and self._path_exists(source_data_path):
            source_df = self.load_source_dataframe(source_data_path)
            if not preflight and query_column in source_df.columns:
                preflight = PreflightService().analyze(
                    source_df,
                    query_col=query_column,
                    pricing=resolve_pricing_config(CONFIG),
                )
        else:
            source_df = self._load_source_dataframe_fallback(record)
            if source_df is not None and not preflight and query_column in source_df.columns:
                preflight = PreflightService().analyze(
                    source_df,
                    query_col=query_column,
                    pricing=resolve_pricing_config(CONFIG),
                )

        run_artifacts = self._load_run_artifacts_from_record(record, report_language=report_language)
        return {
            "record": record,
            "source_df": source_df,
            "query_column": query_column,
            "preflight": preflight,
            "run_artifacts": run_artifacts,
            "session_snapshot": record.get("session_snapshot"),
            "can_resume": record.get("status") in {"cancelled", "failed", "running"},
        }

    def load_run_artifacts(
        self,
        record: Dict[str, Any],
        *,
        report_language: str = "zh",
    ) -> Dict[str, Any]:
        artifacts = self._load_run_artifacts_from_record(record, report_language=report_language)
        if artifacts is None:
            raise FileNotFoundError("This record has no saved result artifacts.")
        return artifacts

    def save_manifest(
        self,
        *,
        dataset_name: str,
        display_name: str,
        timestamp: str,
        run_mode: str,
        summary_path: Optional[Path],
        hierarchy_path: Optional[Path],
        csv_path: Optional[Path],
        agent_report_path: Optional[Path],
        runtime_seconds: Optional[float],
        llm_model: Optional[str],
        embedding_model: Optional[str],
        report_language: Optional[str],
    ) -> Path:
        record = RunRecord(
            run_id=f"{dataset_name}:{timestamp}:{run_mode}",
            dataset_name=dataset_name,
            display_name=display_name,
            timestamp=timestamp,
            run_mode=run_mode,
            status="completed",
            source="manifest",
            summary_path=str(summary_path) if summary_path else None,
            hierarchy_path=str(hierarchy_path) if hierarchy_path else None,
            csv_path=str(csv_path) if csv_path else None,
            agent_report_path=str(agent_report_path) if agent_report_path else None,
            runtime_seconds=runtime_seconds,
            llm_model=llm_model,
            embedding_model=embedding_model,
            report_language=report_language,
        )
        return self.upsert_manifest(asdict(record))

    def _load_run_artifacts_from_record(
        self,
        record: Dict[str, Any],
        *,
        report_language: str,
    ) -> Optional[Dict[str, Any]]:
        summary_path_value = record.get("summary_path")
        if not summary_path_value or not self._path_exists(summary_path_value):
            return None

        summary_path = self._resolve_path(summary_path_value)
        summary = self.report_service.load_summary(summary_path)
        hierarchy_path_value = record.get("hierarchy_path")
        hierarchy_path = self._resolve_optional_path(hierarchy_path_value)
        hierarchy = self.report_service.load_hierarchy(hierarchy_path) if hierarchy_path else None
        csv_path = self._resolve_optional_path(record.get("csv_path"))
        agent_report_path = self._resolve_optional_path(record.get("agent_report_path"))

        if agent_report_path and agent_report_path.exists():
            agent_report_text = agent_report_path.read_text(encoding="utf-8")
        else:
            agent_report_path = self._build_cached_report_path(
                dataset_name=record.get("dataset_name") or summary.get("dataset", "unknown"),
                timestamp=record.get("timestamp") or summary.get("timestamp", "unknown"),
                report_language=report_language,
            )
            agent_report_text = self.report_service.build_agent_report(
                summary=summary,
                hierarchy=hierarchy,
                language=report_language,
                run_metadata={
                    "run_mode": record.get("run_mode", "loaded"),
                    "display_name": record.get("display_name") or summary.get("display_name"),
                    "runtime_seconds": record.get("runtime_seconds", "cached"),
                    "llm_model": record.get("llm_model", "cached"),
                    "embedding_model": record.get("embedding_model", "cached"),
                },
                enable_llm_summary=False,
            )
            agent_report_path.write_text(agent_report_text, encoding="utf-8")

        return {
            "summary": summary,
            "hierarchy": hierarchy,
            "summary_path": summary_path,
            "hierarchy_path": hierarchy_path,
            "csv_path": csv_path,
            "agent_report_path": agent_report_path,
            "agent_report_text": agent_report_text,
            "run_mode": record.get("run_mode", "loaded"),
            "runtime_seconds": record.get("runtime_seconds", "cached"),
            "display_name": record.get("display_name") or summary.get("display_name"),
            "history_run_id": record.get("run_id"),
        }

    def _load_legacy_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        hierarchy_map = self._build_hierarchy_source_map()

        for summary_path in sorted(self.output_dir.glob("*_summary_*.json"), reverse=True):
            try:
                summary = self.report_service.load_summary(summary_path)
            except Exception:
                continue

            resolved_summary = str(summary_path.resolve())
            hierarchy_path = hierarchy_map.get(resolved_summary)
            dataset_name = summary.get("dataset") or summary_path.stem.split("_summary_")[0]
            timestamp = summary.get("timestamp") or self._extract_timestamp(summary_path.name)
            records.append(
                asdict(
                    RunRecord(
                        run_id=f"legacy:{summary_path.name}",
                        dataset_name=dataset_name,
                        display_name=summary.get("display_name") or dataset_name,
                        timestamp=timestamp,
                        run_mode="legacy",
                        status="completed",
                        source="legacy",
                        summary_path=str(summary_path),
                        hierarchy_path=str(hierarchy_path) if hierarchy_path else None,
                    )
                )
            )

        return records

    def _build_hierarchy_source_map(self) -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        for hierarchy_path in self.output_dir.glob("*_hierarchy_*.json"):
            try:
                hierarchy = self.report_service.load_hierarchy(hierarchy_path)
            except Exception:
                continue

            source_summary_path = hierarchy.get("source_summary_path")
            if not source_summary_path:
                continue
            mapping[str(self._resolve_path(source_summary_path))] = hierarchy_path
        return mapping

    def _build_cached_report_path(
        self,
        *,
        dataset_name: str,
        timestamp: str,
        report_language: str,
    ) -> Path:
        return self.output_dir / f"{dataset_name}_agent_report_cached_{timestamp}_{report_language}.md"

    def _load_source_dataframe_fallback(self, record: Dict[str, Any]) -> Optional[pd.DataFrame]:
        dataset_name = record.get("dataset_name")
        display_name = record.get("display_name")
        candidates = [name for name in [dataset_name, display_name] if name]

        for base_name in candidates:
            for extension in PreflightService.SUPPORTED_EXTENSIONS:
                candidate = self.raw_dir / f"{base_name}{extension}"
                if not candidate.exists():
                    continue
                if extension == ".csv":
                    return pd.read_csv(candidate)
                return pd.read_excel(candidate)
        return None

    def _extract_timestamp(self, filename: str) -> str:
        stem = Path(filename).stem
        parts = stem.rsplit("_", 2)
        if len(parts) >= 2:
            return "_".join(parts[-2:])
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _resolve_optional_path(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        return self._resolve_path(value)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return path if path.exists() else (Path.cwd() / path)

    def _path_exists(self, value: Optional[str]) -> bool:
        if not value:
            return False
        return self._resolve_path(value).exists()
