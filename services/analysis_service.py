"""
Reusable analysis entrypoints for CLI and local UI.
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from config.config_loader import CONFIG
from core.graph import TDRClusterGraph
from services.dataset_manager import DatasetManager
from services.graph_state_service import GraphStateService
from services.hierarchy_service import HierarchyService
from services.llm_service import LLMService
from services.report_service import ReportService
from services.run_control import CancelCallback, check_cancelled


EventCallback = Optional[Callable[[str, Dict[str, Any]], None]]
LogCallback = Optional[Callable[[str], None]]


@dataclass
class RunArtifacts:
    dataset_name: str
    display_name: str
    run_mode: str
    csv_path: Optional[Path]
    summary_path: Path
    hierarchy_path: Optional[Path]
    agent_report_path: Path
    manifest_path: Optional[Path]
    summary: Dict[str, Any]
    hierarchy: Optional[Dict[str, Any]]
    runtime_seconds: float


class _CallbackLogStream(io.TextIOBase):
    def __init__(self, callback: LogCallback, mirror_stream: io.TextIOBase):
        self.callback = callback
        self.mirror_stream = mirror_stream
        self._buffer = ""

    def write(self, text: str) -> int:
        if self.mirror_stream:
            self.mirror_stream.write(text)
            self.mirror_stream.flush()

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            normalized = line.rstrip()
            if normalized and self.callback:
                self.callback(normalized)
        return len(text)

    def flush(self):
        if self._buffer.strip() and self.callback:
            self.callback(self._buffer.strip())
        self._buffer = ""
        if self.mirror_stream:
            self.mirror_stream.flush()


class AnalysisService:
    """Programmatic surface for IntentAtlas runs."""

    def __init__(self):
        self.report_service = ReportService()

    def persist_dataframe_as_dataset(self, df: pd.DataFrame, dataset_name: str) -> Path:
        raw_dir = Path("data/raw_data")
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_path = raw_dir / f"{dataset_name}.csv"
        df.to_csv(output_path, index=False)
        return output_path

    def build_preview_dataset_name(self, dataset_name: str) -> str:
        return f"{dataset_name}_preview"

    def sanitize_dataset_name(self, value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_analysis(
        self,
        *,
        dataset_name: str,
        display_name: Optional[str] = None,
        sample_size: Optional[int] = None,
        run_mode_override: Optional[str] = None,
        report_language: str = "zh",
        output_dir: str = "output",
        build_hierarchy: bool = True,
        enable_llm_report_summary: bool = True,
        event_callback: EventCallback = None,
        log_callback: LogCallback = None,
        should_cancel: CancelCallback = None,
        run_id: Optional[str] = None,
        resume_state_path: Optional[str] = None,
    ) -> RunArtifacts:
        start_time = time.time()
        run_mode = run_mode_override or ("preview" if sample_size else "full")
        display_name = display_name or dataset_name
        manifest_path = None
        checkpoint_service = GraphStateService(output_dir=output_dir)
        checkpoint_path = Path(resume_state_path) if resume_state_path else None

        check_cancelled(should_cancel)
        self._emit(event_callback, "stage_started", stage="loading", label="Loading dataset")
        llm_service = LLMService(CONFIG.get("llm", {}))
        dataset_manager = DatasetManager(observer=event_callback, should_cancel=should_cancel)

        log_stream = _CallbackLogStream(log_callback, sys.__stdout__)
        with redirect_stdout(log_stream):
            initial_graph_state = None
            queries = None
            if checkpoint_path and checkpoint_path.exists():
                initial_graph_state = checkpoint_service.load_state(checkpoint_path)
                self._emit(
                    event_callback,
                    "graph_state_restored",
                    run_id=run_id,
                    checkpoint_path=str(checkpoint_path),
                )
            else:
                queries = dataset_manager.load_dataset_as_queries(dataset_name, sample_size)

            check_cancelled(should_cancel)
            self._emit(
                event_callback,
                "stage_started",
                stage="taxonomy",
                label="Running taxonomy induction",
                total_queries=initial_graph_state["total_queries"] if initial_graph_state else len(queries),
                run_mode=run_mode,
            )

            def checkpoint_callback(graph_state):
                if not run_id:
                    return
                path = checkpoint_service.save_state(run_id, graph_state)
                self._emit(
                    event_callback,
                    "graph_checkpoint",
                    run_id=run_id,
                    checkpoint_path=str(path),
                )

            tdr_graph = TDRClusterGraph(
                llm_service,
                observer=event_callback,
                should_cancel=should_cancel,
                checkpoint_callback=checkpoint_callback,
            )
            final_state = tdr_graph.run(initial_queries=queries, initial_state=initial_graph_state)

            check_cancelled(should_cancel)
            self._emit(event_callback, "stage_started", stage="saving", label="Saving raw artifacts")
            csv_path, summary_path = save_results(
                final_state=final_state,
                output_dir=output_dir,
                dataset_name=dataset_name,
                display_name=display_name,
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            hierarchy = None
            hierarchy_path = None

            if build_hierarchy and summary.get("categories"):
                check_cancelled(should_cancel)
                self._emit(event_callback, "stage_started", stage="hierarchy", label="Building hierarchy")
                hierarchy_service = HierarchyService(llm_service)
                hierarchy = hierarchy_service.build_hierarchy_from_summary(summary)
                hierarchy["source_summary_path"] = str(summary_path)
                hierarchy_path = default_hierarchy_path(summary_path, dataset_name)
                hierarchy_path.write_text(
                    json.dumps(hierarchy, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            runtime_seconds = round(time.time() - start_time, 2)
            report_name = f"{dataset_name}_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            agent_report_path = Path(output_dir) / report_name

            check_cancelled(should_cancel)
            self._emit(event_callback, "stage_started", stage="report", label="Generating agent report")
            self.report_service.save_agent_report(
                output_path=agent_report_path,
                summary=summary,
                hierarchy=hierarchy,
                language=report_language,
                llm_service=llm_service,
                enable_llm_summary=enable_llm_report_summary,
                run_metadata={
                    "run_mode": run_mode,
                    "display_name": display_name,
                    "runtime_seconds": runtime_seconds,
                    "llm_model": CONFIG.get("llm", {}).get("model_name", "unknown"),
                    "embedding_model": CONFIG.get("embedding", {}).get("model_name", "unknown"),
                    "sample_size": sample_size,
                    "build_hierarchy": build_hierarchy,
                },
            )
            if run_id:
                manifest_path = checkpoint_service.output_dir / f"run_manifest_{run_id}.json"
                checkpoint_service.clear_state(checkpoint_service.checkpoint_path(run_id))
            elif checkpoint_path:
                checkpoint_service.clear_state(checkpoint_path)

        runtime_seconds = round(time.time() - start_time, 2)
        self._emit(
            event_callback,
            "run_completed",
            stage="completed",
            runtime_seconds=runtime_seconds,
            dataset_name=dataset_name,
            summary_path=str(summary_path),
            hierarchy_path=str(hierarchy_path) if hierarchy_path else None,
            agent_report_path=str(agent_report_path),
        )
        return RunArtifacts(
            dataset_name=dataset_name,
            display_name=display_name,
            run_mode=run_mode,
            csv_path=csv_path,
            summary_path=summary_path,
            hierarchy_path=hierarchy_path,
            agent_report_path=agent_report_path,
            manifest_path=manifest_path,
            summary=summary,
            hierarchy=hierarchy,
            runtime_seconds=runtime_seconds,
        )

    def _emit(self, callback: EventCallback, event: str, **payload):
        if callback:
            callback(event, payload)


def default_hierarchy_path(summary_path: Path, dataset_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return summary_path.parent / f"{dataset_name}_hierarchy_{timestamp}.json"


def save_results(
    final_state: dict,
    output_dir: str,
    dataset_name: str,
    display_name: Optional[str] = None,
) -> tuple[Optional[Path], Path]:
    """Save clustering assignments and summary for both CLI and UI flows."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"💾 保存结果到: {output_dir}/")

    categories = final_state.get("categories", {})
    residual_clusters = final_state.get("residual_clusters", [])

    results_data = []
    for category in categories.values():
        for query in category.queries:
            results_data.append(
                {
                    "query_id": query.id,
                    "query_content": query.content,
                    "category_id": category.id,
                    "category_name": category.name,
                    "category_definition": category.definition,
                    "boundary_notes": category.boundary_notes,
                    "dataset": dataset_name,
                    "timestamp": timestamp,
                }
            )

    csv_path = None
    if results_data:
        results_df = pd.DataFrame(results_data)
        csv_path = output_path / f"{dataset_name}_clustering_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"   ✓ 详细结果: {csv_path}")

    summary_data = {
        "dataset": dataset_name,
        "display_name": display_name or dataset_name,
        "timestamp": timestamp,
        "total_queries": sum(len(cat.queries) for cat in categories.values()) + sum(
            len(cluster.queries) for cluster in residual_clusters
        ),
        "classified_query_count": sum(len(cat.queries) for cat in categories.values()),
        "total_categories": len(categories),
        "residual_cluster_count": len(residual_clusters),
        "residual_query_count": sum(len(cluster.queries) for cluster in residual_clusters),
        "categories": {},
        "residual_clusters": {},
    }

    for cat_id, category in categories.items():
        summary_data["categories"][cat_id] = {
            "name": category.name,
            "definition": category.definition,
            "boundary_notes": category.boundary_notes,
            "query_count": len(category.queries),
            "positive_examples": category.positive_examples[:5],
        }

    for cluster in residual_clusters[:50]:
        summary_data["residual_clusters"][cluster.id] = {
            "query_count": len(cluster.queries),
            "depth": cluster.depth,
            "is_noise": cluster.is_noise,
            "sample_texts": cluster.sample_texts[:5],
            "evidence_rounds": cluster.evidence_rounds,
            "reason": (cluster.judge or {}).get("residual_reason", "unspecified"),
        }

    json_path = output_path / f"{dataset_name}_summary_{timestamp}.json"
    json_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   ✓ 摘要报告: {json_path}")

    return csv_path, json_path
