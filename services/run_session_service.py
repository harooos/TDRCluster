"""
Background run manager with persisted project manifests.
"""
from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from config.config_loader import CONFIG
from services.analysis_service import AnalysisService
from services.run_control import RunCancelledError
from services.run_history_service import RunHistoryService


@dataclass
class RunSession:
    run_id: str
    dataset_name: str
    display_name: str
    run_mode: str
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: Optional[str] = None
    stop_requested: bool = False
    error: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    activity: list[Dict[str, str]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(
        default_factory=lambda: {
            "reviewed": 0,
            "create": 0,
            "merge": 0,
            "split": 0,
            "inspect": 0,
            "categories": 0,
            "residual": 0,
            "pending_tasks": 0,
            "pending_microclusters": 0,
            "embedding_completed": 0,
            "embedding_total": 0,
            "total_queries": 0,
        }
    )


class RunSessionService:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, RunSession] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self.history_service = RunHistoryService()

    def start_run(
        self,
        *,
        dataset_name: str,
        display_name: str,
        run_mode: str,
        sample_size: Optional[int],
        report_language: str,
        build_hierarchy: bool,
        enable_llm_report_summary: bool = True,
        output_dir: str = "output",
        run_id: Optional[str] = None,
        source_df: Optional[pd.DataFrame] = None,
        source_data_path: Optional[str] = None,
        source_row_count: Optional[int] = None,
        uploaded_name: Optional[str] = None,
        query_column: str = "query",
        preflight: Optional[Dict[str, Any]] = None,
        preview_rows: Optional[int] = None,
        resume_state_path: Optional[str] = None,
    ) -> str:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        manifest = self.history_service.load_manifest(run_id) or {}
        session = self._session_from_snapshot(
            manifest.get("session_snapshot"),
            run_id=run_id,
            dataset_name=dataset_name,
            display_name=display_name,
            run_mode=run_mode,
        )
        session.status = "running"
        session.stage = "queued"
        session.progress = max(1, session.progress)
        session.finished_at = None
        session.stop_requested = False
        session.error = None
        session.artifacts = None
        session.activity.append(self._activity("info", self._friendly_message("run_started", {"display_name": display_name})))

        cancel_event = threading.Event()
        with self._lock:
            self._sessions[run_id] = session
            self._cancel_events[run_id] = cancel_event

        if source_df is not None and source_data_path is None:
            source_data_path = str(self.history_service.save_source_dataframe(run_id, source_df))
            source_row_count = len(source_df)

        timestamp = manifest.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._persist_manifest(
            run_id,
            {
                "run_id": run_id,
                "dataset_name": dataset_name,
                "display_name": display_name,
                "timestamp": timestamp,
                "run_mode": run_mode,
                "status": "running",
                "source": "manifest",
                "report_language": report_language,
                "llm_model": CONFIG.get("llm", {}).get("model_name") or manifest.get("llm_model"),
                "embedding_model": CONFIG.get("embedding", {}).get("model_name") or manifest.get("embedding_model"),
                "source_data_path": source_data_path or manifest.get("source_data_path"),
                "source_row_count": source_row_count or manifest.get("source_row_count"),
                "uploaded_name": uploaded_name or manifest.get("uploaded_name"),
                "query_column": query_column or manifest.get("query_column") or "query",
                "preview_rows": preview_rows if preview_rows is not None else manifest.get("preview_rows"),
                "build_hierarchy": build_hierarchy,
                "preflight": preflight or manifest.get("preflight"),
                "resume_state_path": resume_state_path or manifest.get("resume_state_path"),
            },
        )

        thread = threading.Thread(
            target=self._run_worker,
            kwargs={
                "run_id": run_id,
                "dataset_name": dataset_name,
                "display_name": display_name,
                "run_mode": run_mode,
                "sample_size": sample_size,
                "report_language": report_language,
                "build_hierarchy": build_hierarchy,
                "enable_llm_report_summary": enable_llm_report_summary,
                "output_dir": output_dir,
                "resume_state_path": resume_state_path or manifest.get("resume_state_path"),
            },
            daemon=True,
        )
        with self._lock:
            self._threads[run_id] = thread
        thread.start()
        return run_id

    def register_persisted_session(
        self,
        run_id: str,
        snapshot: Optional[Dict[str, Any]],
        *,
        dataset_name: str,
        display_name: str,
        run_mode: str,
    ):
        session = self._session_from_snapshot(
            snapshot,
            run_id=run_id,
            dataset_name=dataset_name,
            display_name=display_name,
            run_mode=run_mode,
        )
        with self._lock:
            self._sessions[run_id] = session
            self._cancel_events.setdefault(run_id, threading.Event())

    def request_stop(self, run_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(run_id)
            cancel_event = self._cancel_events.get(run_id)
            if not session or not cancel_event or session.status not in {"queued", "running"}:
                return False
            session.stop_requested = True
            session.activity.append(self._activity("warning", "已收到停止请求，正在安全结束当前步骤。"))
            cancel_event.set()
            self._persist_manifest_locked(run_id, {"status": session.status, "session_snapshot": self._persisted_snapshot(session)})
            return True

    def get_session(self, run_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not run_id:
            return None
        with self._lock:
            session = self._sessions.get(run_id)
            if not session:
                return None
            return self._snapshot(session)

    def _run_worker(
        self,
        *,
        run_id: str,
        dataset_name: str,
        display_name: str,
        run_mode: str,
        sample_size: Optional[int],
        report_language: str,
        build_hierarchy: bool,
        enable_llm_report_summary: bool,
        output_dir: str,
        resume_state_path: Optional[str],
    ):
        analysis_service = AnalysisService()
        cancel_event = self._cancel_events[run_id]

        def should_cancel() -> bool:
            return cancel_event.is_set()

        try:
            artifacts = analysis_service.run_analysis(
                dataset_name=dataset_name,
                display_name=display_name,
                sample_size=sample_size,
                run_mode_override=run_mode,
                report_language=report_language,
                build_hierarchy=build_hierarchy,
                enable_llm_report_summary=enable_llm_report_summary,
                event_callback=lambda event, payload: self._handle_event(run_id, event, payload),
                should_cancel=should_cancel,
                output_dir=output_dir,
                run_id=run_id,
                resume_state_path=resume_state_path,
            )
            with self._lock:
                session = self._sessions[run_id]
                session.status = "completed"
                session.stage = "completed"
                session.progress = 100
                session.finished_at = datetime.now().isoformat(timespec="seconds")
                session.artifacts = {
                    "dataset_name": artifacts.dataset_name,
                    "display_name": artifacts.display_name,
                    "run_mode": artifacts.run_mode,
                    "csv_path": str(artifacts.csv_path) if artifacts.csv_path else None,
                    "summary_path": str(artifacts.summary_path),
                    "hierarchy_path": str(artifacts.hierarchy_path) if artifacts.hierarchy_path else None,
                    "agent_report_path": str(artifacts.agent_report_path),
                    "manifest_path": str(artifacts.manifest_path) if artifacts.manifest_path else None,
                    "summary": artifacts.summary,
                    "hierarchy": artifacts.hierarchy,
                    "runtime_seconds": artifacts.runtime_seconds,
                }
                session.activity.append(self._activity("success", "分析已完成，结果已经可以查看。"))
                self._persist_manifest_locked(
                    run_id,
                    {
                        "status": "completed",
                        "runtime_seconds": artifacts.runtime_seconds,
                        "summary_path": str(artifacts.summary_path),
                        "hierarchy_path": str(artifacts.hierarchy_path) if artifacts.hierarchy_path else None,
                        "csv_path": str(artifacts.csv_path) if artifacts.csv_path else None,
                        "agent_report_path": str(artifacts.agent_report_path),
                        "resume_state_path": None,
                        "session_snapshot": self._persisted_snapshot(session),
                    },
                )
        except RunCancelledError:
            with self._lock:
                session = self._sessions[run_id]
                session.status = "cancelled"
                session.stage = "cancelled"
                session.finished_at = datetime.now().isoformat(timespec="seconds")
                session.activity.append(self._activity("warning", "运行已停止。"))
                self._persist_manifest_locked(
                    run_id,
                    {
                        "status": "cancelled",
                        "session_snapshot": self._persisted_snapshot(session),
                    },
                )
        except Exception as exc:
            with self._lock:
                session = self._sessions[run_id]
                session.status = "failed"
                session.stage = "failed"
                session.finished_at = datetime.now().isoformat(timespec="seconds")
                session.error = str(exc)
                session.activity.append(self._activity("error", f"运行失败：{exc}"))
                self._persist_manifest_locked(
                    run_id,
                    {
                        "status": "failed",
                        "session_snapshot": self._persisted_snapshot(session),
                    },
                )

    def _handle_event(self, run_id: str, event: str, payload: Dict[str, Any]):
        with self._lock:
            session = self._sessions.get(run_id)
            if not session:
                return

            if event == "run_started":
                session.metrics["total_queries"] = payload.get("total_queries", 0)
                self._append_activity(session, "info", self._friendly_message(event, payload))
            elif event == "stage_started":
                stage = payload.get("stage", "running")
                session.stage = stage
                session.progress = self._stage_progress(stage)
                self._append_activity(session, "info", self._friendly_message(stage, payload))
            elif event == "embedding_started":
                session.stage = "embedding"
                session.progress = 12
                session.metrics["embedding_total"] = payload.get("total_rows", 0)
                self._append_activity(session, "info", self._friendly_message(event, payload))
            elif event == "embedding_reused":
                total_rows = payload.get("total_rows", session.metrics["embedding_total"])
                session.metrics["embedding_total"] = total_rows
                session.metrics["embedding_completed"] = total_rows
                self._append_activity(session, "info", self._friendly_message(event, payload))
            elif event == "embedding_progress":
                total_rows = payload.get("total_rows", session.metrics["embedding_total"])
                completed_rows = payload.get("completed_rows", 0)
                session.stage = "embedding"
                session.metrics["embedding_total"] = total_rows
                session.metrics["embedding_completed"] = completed_rows
                ratio = completed_rows / max(1, total_rows)
                session.progress = min(30, max(12, int(12 + ratio * 18)))
                self._append_activity(session, "info", self._friendly_message(event, payload))
            elif event == "graph_state_restored":
                self._append_activity(session, "info", "已从上次停止的位置继续运行。")
                self._persist_manifest_locked(run_id, {"resume_state_path": payload.get("checkpoint_path")})
            elif event == "graph_checkpoint":
                self._persist_manifest_locked(run_id, {"resume_state_path": payload.get("checkpoint_path")})
            elif event == "reviewer_completed":
                action = (payload.get("action") or "").lower()
                session.metrics["reviewed"] += 1
                if action in session.metrics:
                    session.metrics[action] += 1
                self._append_activity(session, "info", self._friendly_message(event, payload))
            elif event == "splitter_completed":
                self._append_activity(session, "info", self._friendly_message(event, payload))
            elif event == "inspect_completed":
                self._append_activity(session, "info", self._friendly_message(event, payload))
            elif event == "dispatcher_completed":
                session.metrics["categories"] = payload.get("category_count", session.metrics["categories"])
                session.metrics["residual"] = payload.get("residual_count", session.metrics["residual"])
                session.metrics["pending_tasks"] = payload.get("pending_tasks", session.metrics["pending_tasks"])
                session.metrics["pending_microclusters"] = payload.get("pending_microclusters", session.metrics["pending_microclusters"])
                session.progress = max(session.progress, self._taxonomy_progress(session.metrics))
            elif event == "run_completed":
                session.progress = 100

            self._persist_manifest_locked(run_id, {"session_snapshot": self._persisted_snapshot(session)})

    def _persist_manifest(self, run_id: str, patch: Dict[str, Any]):
        with self._lock:
            self._persist_manifest_locked(run_id, patch)

    def _persist_manifest_locked(self, run_id: str, patch: Dict[str, Any]):
        existing = self.history_service.load_manifest(run_id) or {}
        payload = existing.copy()
        payload.update(patch)
        payload["run_id"] = run_id
        self.history_service.upsert_manifest(payload)

    def _snapshot(self, session: RunSession) -> Dict[str, Any]:
        return {
            "run_id": session.run_id,
            "dataset_name": session.dataset_name,
            "display_name": session.display_name,
            "run_mode": session.run_mode,
            "status": session.status,
            "stage": session.stage,
            "progress": session.progress,
            "created_at": session.created_at,
            "finished_at": session.finished_at,
            "stop_requested": session.stop_requested,
            "error": session.error,
            "artifacts": deepcopy(session.artifacts),
            "activity": list(session.activity),
            "metrics": deepcopy(session.metrics),
        }

    def _persisted_snapshot(self, session: RunSession) -> Dict[str, Any]:
        snapshot = self._snapshot(session)
        artifacts = snapshot.get("artifacts") or {}
        snapshot["artifacts"] = {
            "dataset_name": artifacts.get("dataset_name"),
            "display_name": artifacts.get("display_name"),
            "run_mode": artifacts.get("run_mode"),
            "csv_path": artifacts.get("csv_path"),
            "summary_path": artifacts.get("summary_path"),
            "hierarchy_path": artifacts.get("hierarchy_path"),
            "agent_report_path": artifacts.get("agent_report_path"),
            "manifest_path": artifacts.get("manifest_path"),
            "runtime_seconds": artifacts.get("runtime_seconds"),
        } if artifacts else None
        return snapshot

    def _session_from_snapshot(
        self,
        snapshot: Optional[Dict[str, Any]],
        *,
        run_id: str,
        dataset_name: str,
        display_name: str,
        run_mode: str,
    ) -> RunSession:
        if not snapshot:
            return RunSession(
                run_id=run_id,
                dataset_name=dataset_name,
                display_name=display_name,
                run_mode=run_mode,
                status="queued",
                stage="queued",
                progress=0,
            )
        return RunSession(
            run_id=run_id,
            dataset_name=snapshot.get("dataset_name", dataset_name),
            display_name=snapshot.get("display_name", display_name),
            run_mode=snapshot.get("run_mode", run_mode),
            status=snapshot.get("status", "queued"),
            stage=snapshot.get("stage", "queued"),
            progress=int(snapshot.get("progress", 0)),
            created_at=snapshot.get("created_at") or datetime.now().isoformat(timespec="seconds"),
            finished_at=snapshot.get("finished_at"),
            stop_requested=bool(snapshot.get("stop_requested", False)),
            error=snapshot.get("error"),
            artifacts=deepcopy(snapshot.get("artifacts")),
            activity=list(snapshot.get("activity", [])),
            metrics=deepcopy(snapshot.get("metrics", RunSession(run_id, dataset_name, display_name, run_mode).metrics)),
        )

    def _append_activity(self, session: RunSession, kind: str, message: str):
        if not message:
            return
        if session.activity and session.activity[-1]["message"] == message:
            return
        session.activity.append(self._activity(kind, message))
        if len(session.activity) > 24:
            session.activity = session.activity[-24:]

    def _activity(self, kind: str, message: str) -> Dict[str, str]:
        return {
            "kind": kind,
            "message": message,
            "time": datetime.now().strftime("%H:%M:%S"),
        }

    def _stage_progress(self, stage: str) -> int:
        return {
            "loading": 6,
            "embedding": 18,
            "taxonomy": 42,
            "saving": 84,
            "hierarchy": 92,
            "report": 97,
            "completed": 100,
        }.get(stage, 5)

    def _taxonomy_progress(self, metrics: Dict[str, Any]) -> int:
        reviewed = int(metrics.get("reviewed", 0))
        pending_microclusters = int(metrics.get("pending_microclusters", 0))
        pending_tasks = int(metrics.get("pending_tasks", 0))
        total_work = reviewed + pending_microclusters + pending_tasks
        if total_work <= 0:
            return 42
        ratio = reviewed / max(1, total_work)
        return min(82, max(42, int(42 + ratio * 40)))

    def _friendly_message(self, key: str, payload: Dict[str, Any]) -> str:
        if key == "run_started":
            return "已开始分析。"
        if key == "loading":
            return "正在准备数据。"
        if key == "taxonomy":
            return "正在归纳主题地图。"
        if key == "saving":
            return "正在整理并保存结果。"
        if key == "hierarchy":
            return "正在生成层级结构。"
        if key == "report":
            return "正在生成最终报告。"
        if key == "embedding_started":
            return f"正在生成 embedding，共 {payload.get('total_rows', 0)} 条文本。"
        if key == "embedding_reused":
            return "检测到已有 embedding 缓存，直接复用。"
        if key == "embedding_progress":
            return f"Embedding 进度 {payload.get('completed_rows', 0)}/{payload.get('total_rows', 0)}。"
        if key == "splitter_completed":
            return (
                f"完成一轮物理分组，得到 {payload.get('review_cluster_count', 0)} 个待判断主题，"
                f"并识别 {payload.get('residual_cluster_count', 0)} 个待确认片段。"
            )
        if key == "reviewer_completed":
            action = (payload.get("action") or "").lower()
            return {
                "create": "发现一个新主题。",
                "merge": "一个片段已并入已有主题。",
                "split": "一个片段被判定为仍需继续细分。",
                "inspect": "模型要求补充更多样本后再判断。",
            }.get(action, "完成一次主题判断。")
        if key == "inspect_completed":
            if payload.get("forced_final"):
                return "补充样本后已达到停止条件，下一轮将直接做最终判断。"
            return "已补充更多代表样本，继续判断。"
        return ""


RUN_SESSION_SERVICE = RunSessionService()
