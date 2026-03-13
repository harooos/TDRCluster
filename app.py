from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from services.analysis_service import AnalysisService
from services.llm_service import LLMService
from services.preflight_service import PreflightService, resolve_pricing_config
from services.report_service import ReportService
from services.run_history_service import RunHistoryService
from services.run_session_service import RUN_SESSION_SERVICE
from services.runtime_config import (
    apply_runtime_config,
    load_and_apply_config,
    load_effective_config,
    save_config,
)


UI_TEXT = {
    "zh": {
        "page_title": "IntentAtlas",
        "caption": "把原始 query 转成可交互的意图地图。",
        "settings": "设置",
        "language": "界面语言",
        "model_config": "模型配置",
        "llm_base_url": "LLM Base URL",
        "llm_api_key": "LLM API Key",
        "llm_model": "LLM 模型",
        "embedding_base_url": "Embedding Base URL",
        "embedding_api_key": "Embedding API Key",
        "embedding_api_style": "Embedding 接口风格",
        "embedding_endpoint": "Embedding Endpoint Path",
        "embedding_model": "Embedding 模型",
        "taxonomy_context": "可选业务上下文",
        "taxonomy_context_help": "可选的业务背景信息，会用于主题判断阶段。",
        "cost_estimation": "成本估算",
        "embedding_price": "Embedding 每百万输入 token 价格（元）",
        "llm_input_price": "LLM 每百万输入 token 价格（元）",
        "llm_output_price": "LLM 每百万输出 token 价格（元）",
        "pricing_defaults_hint": "会根据当前模型自动带入默认价格，你也可以手动覆盖。",
        "save_settings": "保存配置",
        "reload_config": "从磁盘重新加载配置",
        "saved_config": "已保存到 config/config.yaml",
        "saved_runs": "历史结果",
        "saved_runs_help": "这里会列出 output 中已保存的运行结果，随时可以重新打开查看。",
        "select_saved_run": "选择历史结果",
        "load_saved_run": "打开所选结果",
        "refresh_saved_runs": "刷新列表",
        "no_saved_runs": "当前还没有可加载的历史结果。",
        "saved_run_loaded": "已加载历史结果：{label}",
        "saved_run_legacy": "历史导入",
        "project_restored": "已恢复项目：{label}",
        "tab_upload": "上传与预检查",
        "tab_run": "运行",
        "tab_report": "结果报告",
        "upload_title": "上传数据集",
        "upload_label": "拖入 CSV 或 Excel 文件",
        "upload_hint": "先上传数据开始新的分析；历史项目可以从侧边栏重新打开，并恢复到当前工作状态。",
        "parse_failed": "文件解析失败。",
        "query_column": "Query 列",
        "project_name": "项目名称",
        "rows": "总行数",
        "valid_queries": "有效 query",
        "empty_rows": "空行",
        "duplicate_rows": "重复行",
        "embedding_tokens": "Embedding token 估算",
        "suggested_preview": "建议 preview 数",
        "embedding_estimate": "Embedding 估算",
        "llm_estimate": "LLM 估算",
        "total_estimate": "总成本估算",
        "recommended_preview_estimate": "推荐 Preview",
        "full_run_estimate": "全量运行",
        "estimate_basis_rows": "按 `{rows}` 条 query 估算",
        "estimate_total_cost": "预计总成本：`¥{cost}`",
        "estimate_breakdown": "Embedding `¥{embedding}` · LLM `¥{llm}`",
        "input_tokens": "输入 token：`{tokens}`",
        "estimated_cost": "预计成本：`¥{cost}`",
        "token_range": "`{low}` 到 `{high}` tokens",
        "cost_range": "预计成本：`¥{low}` 到 `¥{high}`",
        "preview_data": "上传数据预览",
        "run_title": "运行 IntentAtlas",
        "run_prereq": "请先完成上传和预检查。",
        "no_valid_queries": "所选列中没有有效 query。",
        "run_mode": "运行模式",
        "preview_mode": "preview",
        "full_mode": "full",
        "preview_rows": "Preview 行数",
        "build_hierarchy": "主题地图收敛后自动生成层级树",
        "config_required": "请先在侧边栏配置 LLM 和 embedding 参数。",
        "run_button": "开始运行 IntentAtlas",
        "stop_run": "停止当前运行",
        "continue_run": "继续当前项目",
        "edit_run_config": "返回运行配置",
        "active_run": "运行控制台",
        "run_in_progress": "当前有一个运行正在后台进行中。",
        "run_cancel_requested": "已请求停止，系统会在安全检查点结束当前运行。",
        "run_cancelled": "运行已停止。",
        "run_failed": "运行失败：{error}",
        "run_can_resume": "这个项目可以继续运行。",
        "run_monitor_hint": "这里展示产品化进度，而不是底层终端日志。",
        "current_focus": "当前阶段",
        "recent_activity": "最近进展",
        "run_status": "状态",
        "status_running": "运行中",
        "status_completed": "已完成",
        "status_cancelled": "已停止",
        "status_failed": "失败",
        "status_queued": "准备中",
        "embedded": "已完成 embedding",
        "themes": "主题数",
        "residual_groups": "待确认分组",
        "remaining_work": "剩余队列",
        "stage_loading": "加载数据",
        "stage_embedding": "生成 embedding",
        "stage_taxonomy": "归纳主题地图",
        "stage_saving": "保存原始结果",
        "stage_hierarchy": "生成层级树",
        "stage_report": "生成 Agent Report",
        "current_stage": "当前阶段：{label}",
        "run_completed": "运行完成。",
        "reviewed": "已审簇数",
        "create": "Create",
        "merge": "Merge",
        "split": "Split",
        "inspect": "Inspect",
        "categories": "主题数",
        "run_success": "已完成 {run_mode} 运行，耗时 {runtime_seconds:.2f} 秒。Agent Report 已保存到 {path}。",
        "report_title": "交互式报告",
        "report_hint": "先运行 IntentAtlas，或者从侧边栏打开一个历史结果。",
        "leaf_categories": "叶子类别数",
        "classified_queries": "已归类 query",
        "residual_queries": "残差 query",
        "runtime_seconds_metric": "耗时（秒）",
        "overview": "概览",
        "categories_tab": "类别",
        "hierarchy_tab": "层级树",
        "residuals_tab": "残差",
        "agent_report_tab": "Agent Report",
        "top_category_snapshot": "头部类别概览",
        "no_categories": "当前运行没有可展示的类别。",
        "definition": "定义",
        "boundary_notes": "边界说明",
        "examples": "样本",
        "no_hierarchy": "当前运行没有生成层级树。",
        "reason": "原因",
        "no_residuals": "当前没有残差簇。",
        "download_agent_report": "下载 Agent Report",
        "regenerate_ai_report": "重新生成 AI 摘要",
        "regenerate_ai_report_help": "用当前 LLM 配置，为当前结果重写顶部的人类可读摘要。",
        "regenerate_ai_report_requires_model": "需要先配置 LLM 参数，才能重生成 AI 摘要。",
        "regenerate_ai_report_success": "已重新生成 Agent Report。",
        "preview_agent_report": "预览 Agent Report",
        "raw_artifacts": "原始结果路径",
        "structured_appendix_preview": "结构化附录预览",
        "structured_appendix_heading": "## 结构化附录",
    },
    "en": {
        "page_title": "IntentAtlas",
        "caption": "Turn raw queries into an interactive intent map.",
        "settings": "Settings",
        "language": "Language",
        "model_config": "Model Configuration",
        "llm_base_url": "LLM base URL",
        "llm_api_key": "LLM API key",
        "llm_model": "LLM model",
        "embedding_base_url": "Embedding base URL",
        "embedding_api_key": "Embedding API key",
        "embedding_api_style": "Embedding API style",
        "embedding_endpoint": "Embedding endpoint path",
        "embedding_model": "Embedding model",
        "taxonomy_context": "Optional business context",
        "taxonomy_context_help": "Optional business context used during theme review.",
        "cost_estimation": "Cost Estimation",
        "embedding_price": "Embedding price / 1M input tokens (RMB)",
        "llm_input_price": "LLM price / 1M input tokens (RMB)",
        "llm_output_price": "LLM price / 1M output tokens (RMB)",
        "pricing_defaults_hint": "Default prices are auto-filled from the current model selection, and can still be overridden.",
        "save_settings": "Save Settings",
        "reload_config": "Reload Config From Disk",
        "saved_config": "Saved to config/config.yaml",
        "saved_runs": "Saved Runs",
        "saved_runs_help": "Browse and reopen previously saved runs from the output directory.",
        "select_saved_run": "Select a saved run",
        "load_saved_run": "Open selected run",
        "refresh_saved_runs": "Refresh",
        "no_saved_runs": "No saved runs were found yet.",
        "saved_run_loaded": "Loaded saved run: {label}",
        "saved_run_legacy": "legacy import",
        "project_restored": "Project restored: {label}",
        "tab_upload": "Upload & Check",
        "tab_run": "Run",
        "tab_report": "Report",
        "upload_title": "Upload a dataset",
        "upload_label": "Drop a CSV or Excel file",
        "upload_hint": "Upload a dataset to start a new analysis. Historical projects can be reopened from the sidebar with their working context restored.",
        "parse_failed": "Failed to parse the uploaded file.",
        "query_column": "Query column",
        "project_name": "Project name",
        "rows": "Rows",
        "valid_queries": "Valid queries",
        "empty_rows": "Empty rows",
        "duplicate_rows": "Duplicate rows",
        "embedding_tokens": "Embedding tokens",
        "suggested_preview": "Suggested preview",
        "embedding_estimate": "Embedding estimate",
        "llm_estimate": "LLM estimate",
        "total_estimate": "Total cost estimate",
        "recommended_preview_estimate": "Recommended Preview",
        "full_run_estimate": "Full Run",
        "estimate_basis_rows": "Estimated for `{rows}` queries",
        "estimate_total_cost": "Estimated total cost: `¥{cost}`",
        "estimate_breakdown": "Embedding `¥{embedding}` · LLM `¥{llm}`",
        "input_tokens": "Input tokens: `{tokens}`",
        "estimated_cost": "Estimated cost: `¥{cost}`",
        "token_range": "`{low}` to `{high}` tokens",
        "cost_range": "Estimated cost: `¥{low}` to `¥{high}`",
        "preview_data": "Preview the uploaded data",
        "run_title": "Run IntentAtlas",
        "run_prereq": "Complete the upload and preflight step first.",
        "no_valid_queries": "No valid queries were found in the selected column.",
        "run_mode": "Run mode",
        "preview_mode": "preview",
        "full_mode": "full",
        "preview_rows": "Preview rows",
        "build_hierarchy": "Build hierarchy after the intent map converges",
        "config_required": "Configure both LLM and embedding credentials in the sidebar before running.",
        "run_button": "Run IntentAtlas",
        "stop_run": "Stop current run",
        "continue_run": "Continue this project",
        "edit_run_config": "Back to run setup",
        "active_run": "Run Console",
        "run_in_progress": "A run is currently executing in the background.",
        "run_cancel_requested": "Stop requested. The system will finish the current safe checkpoint first.",
        "run_cancelled": "Run stopped.",
        "run_failed": "Run failed: {error}",
        "run_can_resume": "This project can continue from its current state.",
        "run_monitor_hint": "This surface shows product-friendly progress instead of raw terminal logs.",
        "current_focus": "Current focus",
        "recent_activity": "Recent activity",
        "run_status": "Status",
        "status_running": "Running",
        "status_completed": "Completed",
        "status_cancelled": "Stopped",
        "status_failed": "Failed",
        "status_queued": "Starting",
        "embedded": "Embeddings ready",
        "themes": "Themes",
        "residual_groups": "Unresolved groups",
        "remaining_work": "Remaining queue",
        "stage_loading": "Loading dataset",
        "stage_embedding": "Generating embeddings",
        "stage_taxonomy": "Building the intent map",
        "stage_saving": "Saving raw artifacts",
        "stage_hierarchy": "Building hierarchy",
        "stage_report": "Generating agent report",
        "current_stage": "Current stage: {label}",
        "run_completed": "Run completed.",
        "reviewed": "Reviewed",
        "create": "Create",
        "merge": "Merge",
        "split": "Split",
        "inspect": "Inspect",
        "categories": "Themes",
        "run_success": "Finished {run_mode} run in {runtime_seconds:.2f}s. Agent report saved to {path}.",
        "report_title": "Interactive Report",
        "report_hint": "Run IntentAtlas or open a saved run from the sidebar to inspect a report here.",
        "leaf_categories": "Leaf categories",
        "classified_queries": "Classified queries",
        "residual_queries": "Residual queries",
        "runtime_seconds_metric": "Runtime (s)",
        "overview": "Overview",
        "categories_tab": "Categories",
        "hierarchy_tab": "Hierarchy",
        "residuals_tab": "Residuals",
        "agent_report_tab": "Agent Report",
        "top_category_snapshot": "Top category snapshot",
        "no_categories": "No categories available for this run.",
        "definition": "Definition",
        "boundary_notes": "Boundary notes",
        "examples": "Examples",
        "no_hierarchy": "No hierarchy generated for this run.",
        "reason": "Reason",
        "no_residuals": "No residual clusters recorded.",
        "download_agent_report": "Download Agent Report",
        "regenerate_ai_report": "Regenerate AI Summary",
        "regenerate_ai_report_help": "Rewrite the human-readable report summary using the current LLM configuration.",
        "regenerate_ai_report_requires_model": "Configure the LLM settings before regenerating the AI summary.",
        "regenerate_ai_report_success": "Agent report regenerated.",
        "preview_agent_report": "Preview Agent Report",
        "raw_artifacts": "Raw artifacts",
        "structured_appendix_preview": "Structured Appendix Preview",
        "structured_appendix_heading": "## Structured Appendix",
    },
}

QUALITY_FLAG_TEXT = {
    "More than 5% of rows are empty after trimming.": {
        "zh": "清洗后空行占比超过 5%。",
        "en": "More than 5% of rows are empty after trimming.",
    },
    "Duplicate query ratio is above 20%; deduplication may improve signal quality.": {
        "zh": "重复 query 占比超过 20%，建议先去重以提升信号质量。",
        "en": "Duplicate query ratio is above 20%; deduplication may improve signal quality.",
    },
    "The long-tail query length is high; consider isolating verbose records or transcripts.": {
        "zh": "长尾 query 的长度较高，建议考虑分离超长记录或对话转写文本。",
        "en": "The long-tail query length is high; consider isolating verbose records or transcripts.",
    },
    "Dataset is small; preview and full run will be close in quality.": {
        "zh": "数据集较小，preview 和 full 的效果差异通常不会太大。",
        "en": "Dataset is small; preview and full run will be close in quality.",
    },
    "No valid rows found in the selected query column.": {
        "zh": "所选 query 列中没有有效数据。",
        "en": "No valid rows found in the selected query column.",
    },
}


def current_language() -> str:
    return st.session_state.get("ui_language", "zh")


def tr(key: str, **kwargs) -> str:
    language = current_language()
    template = UI_TEXT.get(language, UI_TEXT["zh"]).get(key, key)
    return template.format(**kwargs) if kwargs else template


def translate_quality_flag(flag: str) -> str:
    mapping = QUALITY_FLAG_TEXT.get(flag)
    if not mapping:
        return flag
    return mapping.get(current_language(), mapping.get("zh", flag))


st.set_page_config(
    page_title="IntentAtlas",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #14213d;
            --muted: #5b6472;
            --accent: #b8501f;
            --accent-soft: rgba(184, 80, 31, 0.12);
            --surface: rgba(255, 253, 247, 0.92);
            --border: rgba(20, 33, 61, 0.10);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(217, 119, 6, 0.16), transparent 24%),
                radial-gradient(circle at top right, rgba(29, 78, 216, 0.10), transparent 26%),
                linear-gradient(180deg, #fffdf8 0%, #f7f3ec 100%);
        }
        .intentatlas-hero {
            padding: 1.6rem 1.8rem;
            border: 1px solid var(--border);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255,255,255,0.90), rgba(255,248,238,0.94));
            box-shadow: 0 16px 48px rgba(20, 33, 61, 0.08);
            margin-bottom: 1.1rem;
        }
        .intentatlas-hero h1 {
            margin: 0;
            color: var(--ink);
            font-size: 2.4rem;
            letter-spacing: -0.04em;
        }
        .intentatlas-hero p {
            margin: 0.45rem 0 0 0;
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 58rem;
        }
        .intentatlas-note {
            padding: 0.9rem 1rem;
            border-left: 4px solid var(--accent);
            background: var(--accent-soft);
            border-radius: 12px;
            color: var(--ink);
            margin: 0.6rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    ensure_session_defaults()
    inject_styles()
    render_sidebar()

    st.markdown(
        f"""
        <div class="intentatlas-hero">
            <h1>IntentAtlas</h1>
            <p>{tr("caption")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([tr("tab_upload"), tr("tab_run"), tr("tab_report")])
    with tabs[0]:
        render_upload_tab()
    with tabs[1]:
        render_run_tab()
    with tabs[2]:
        render_report_tab()


def ensure_session_defaults():
    if "config" not in st.session_state:
        st.session_state["config"] = load_and_apply_config()
    if "ui_language" not in st.session_state:
        st.session_state["ui_language"] = "zh"
    if "uploaded_name" not in st.session_state:
        st.session_state["uploaded_name"] = None
    if "uploaded_bytes" not in st.session_state:
        st.session_state["uploaded_bytes"] = None
    if "uploaded_df" not in st.session_state:
        st.session_state["uploaded_df"] = None
    if "query_col" not in st.session_state:
        st.session_state["query_col"] = None
    if "display_name" not in st.session_state:
        st.session_state["display_name"] = "intentatlas_run"
    if "preflight" not in st.session_state:
        st.session_state["preflight"] = None
    if "run_artifacts" not in st.session_state:
        st.session_state["run_artifacts"] = None
    if "agent_report_text" not in st.session_state:
        st.session_state["agent_report_text"] = None
    if "selected_history_run_id" not in st.session_state:
        st.session_state["selected_history_run_id"] = None
    if "selected_history_run_label" not in st.session_state:
        st.session_state["selected_history_run_label"] = None
    if "loaded_history_run_id" not in st.session_state:
        st.session_state["loaded_history_run_id"] = None
    if "active_run_id" not in st.session_state:
        st.session_state["active_run_id"] = None
    if "synced_run_id" not in st.session_state:
        st.session_state["synced_run_id"] = None
    if "run_mode_selection" not in st.session_state:
        st.session_state["run_mode_selection"] = tr("preview_mode")
    if "preview_rows_selection" not in st.session_state:
        st.session_state["preview_rows_selection"] = None
    if "build_hierarchy_selection" not in st.session_state:
        st.session_state["build_hierarchy_selection"] = True


def render_sidebar():
    st.sidebar.header(tr("settings"))
    current_config = st.session_state["config"]
    pricing = current_config.setdefault("pricing", {})
    language_label = st.sidebar.selectbox(
        tr("language"),
        options=["中文", "English"],
        index=0 if st.session_state["ui_language"] == "zh" else 1,
    )
    st.session_state["ui_language"] = "zh" if language_label == "中文" else "en"

    with st.sidebar.expander(tr("model_config"), expanded=True):
        llm = current_config.setdefault("llm", {})
        embedding = current_config.setdefault("embedding", {})
        runtime = current_config.setdefault("runtime", {})

        llm["base_url"] = st.text_input(tr("llm_base_url"), value=llm.get("base_url", ""), key="llm_base_url")
        llm["api_key"] = st.text_input(tr("llm_api_key"), value=llm.get("api_key", ""), type="password", key="llm_api_key")
        llm["model_name"] = st.text_input(tr("llm_model"), value=llm.get("model_name", ""), key="llm_model_name")

        embedding["base_url"] = st.text_input(tr("embedding_base_url"), value=embedding.get("base_url", ""), key="emb_base_url")
        embedding["api_key"] = st.text_input(tr("embedding_api_key"), value=embedding.get("api_key", ""), type="password", key="emb_api_key")
        embedding["api_style"] = st.selectbox(
            tr("embedding_api_style"),
            options=["openai", "ark_multimodal"],
            index=0 if embedding.get("api_style", "openai") == "openai" else 1,
            key="emb_api_style",
        )
        embedding["endpoint_path"] = st.text_input(
            tr("embedding_endpoint"),
            value=embedding.get("endpoint_path", "/embeddings"),
            key="emb_endpoint_path",
        )
        embedding["model_name"] = st.text_input(
            tr("embedding_model"),
            value=embedding.get("model_name", ""),
            key="emb_model_name",
        )

        runtime["taxonomy_context"] = st.text_area(
            tr("taxonomy_context"),
            value=runtime.get("taxonomy_context") or "",
            help=tr("taxonomy_context_help"),
            key="taxonomy_context",
        )

    with st.sidebar.expander(tr("cost_estimation"), expanded=False):
        resolved_pricing = resolve_pricing_config(current_config)
        st.caption(tr("pricing_defaults_hint"))
        pricing["embedding_input_per_million_tokens"] = st.number_input(
            tr("embedding_price"),
            min_value=0.0,
            value=float(resolved_pricing.embedding_input_per_million_tokens),
            step=0.1,
            format="%.3f",
            key=f"pricing_embedding_{stable_model_key(current_config.get('embedding', {}).get('model_name', ''))}",
        )
        pricing["llm_input_per_million_tokens"] = st.number_input(
            tr("llm_input_price"),
            min_value=0.0,
            value=float(resolved_pricing.llm_input_per_million_tokens),
            step=0.1,
            format="%.3f",
            key=f"pricing_llm_input_{stable_model_key(current_config.get('llm', {}).get('model_name', ''))}",
        )
        pricing["llm_output_per_million_tokens"] = st.number_input(
            tr("llm_output_price"),
            min_value=0.0,
            value=float(resolved_pricing.llm_output_per_million_tokens),
            step=0.1,
            format="%.3f",
            key=f"pricing_llm_output_{stable_model_key(current_config.get('llm', {}).get('model_name', ''))}",
        )
        pricing.pop("embedding_input_per_1k_tokens", None)
        pricing.pop("llm_input_per_1k_tokens", None)
        pricing.pop("llm_output_per_1k_tokens", None)

    if st.sidebar.button(tr("save_settings"), use_container_width=True):
        save_config(current_config)
        apply_runtime_config(current_config)
        st.sidebar.success(tr("saved_config"))

    if st.sidebar.button(tr("reload_config"), use_container_width=True):
        st.session_state["config"] = load_effective_config()
        apply_runtime_config(st.session_state["config"])
        st.rerun()

    st.sidebar.divider()
    render_saved_runs_sidebar()


def render_saved_runs_sidebar():
    st.sidebar.subheader(tr("saved_runs"))
    st.sidebar.caption(tr("saved_runs_help"))

    history_service = RunHistoryService()
    records = history_service.list_runs()
    if not records:
        st.sidebar.info(tr("no_saved_runs"))
        return

    record_map = {record["run_id"]: record for record in records}
    run_ids = list(record_map.keys())
    label_to_run_id: Dict[str, str] = {}
    labels: list[str] = []
    for run_id in run_ids:
        base_label = format_saved_run_label(record_map[run_id])
        label = base_label
        suffix = 2
        while label in label_to_run_id:
            label = f"{base_label} ({suffix})"
            suffix += 1
        label_to_run_id[label] = run_id
        labels.append(label)

    loaded_run_id = st.session_state.get("loaded_history_run_id")
    selected_label = st.session_state.get("selected_history_run_label")
    if selected_label not in label_to_run_id:
        selected_label = next((label for label, run_id in label_to_run_id.items() if run_id == loaded_run_id), labels[0])
    default_index = labels.index(selected_label) if selected_label in labels else 0

    selected_label = st.sidebar.selectbox(
        tr("select_saved_run"),
        options=labels,
        index=default_index,
        key="selected_history_run_label",
    )
    selected_run_id = label_to_run_id.get(selected_label, run_ids[default_index])
    st.session_state["selected_history_run_id"] = selected_run_id

    action_cols = st.sidebar.columns(2)
    if action_cols[0].button(tr("load_saved_run"), use_container_width=True):
        load_saved_project(record_map[selected_run_id])
    if action_cols[1].button(tr("refresh_saved_runs"), use_container_width=True):
        st.rerun()


def render_upload_tab():
    st.subheader(tr("upload_title"))
    st.markdown(
        (
            '<div class="intentatlas-note">'
            + (
                "从这里开始。上传原始 CSV 或 Excel，选择 query 列，让 IntentAtlas 先给出预检查结果、建议 preview 数和成本估算。"
                if current_language() == "zh"
                else "Start here. Upload a raw CSV or Excel file, select the query column, and let IntentAtlas estimate preview size and rough cost before you spend API budget."
            )
            + "</div>"
        ),
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(tr("upload_label"), type=["csv", "xlsx", "xlsm", "xls"])
    preflight_service = PreflightService()

    if uploaded_file is None and st.session_state["uploaded_df"] is None:
        st.info(tr("upload_hint"))
        return

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        if (
            st.session_state["uploaded_name"] != uploaded_file.name
            or st.session_state["uploaded_bytes"] != uploaded_bytes
        ):
            st.session_state["uploaded_name"] = uploaded_file.name
            st.session_state["uploaded_bytes"] = uploaded_bytes
            st.session_state["uploaded_df"] = preflight_service.load_dataframe_from_upload(uploaded_file.name, uploaded_bytes)
            st.session_state["query_col"] = st.session_state["uploaded_df"].columns[0]
            st.session_state["display_name"] = Path(uploaded_file.name).stem
            st.session_state["preflight"] = None
            st.session_state["run_artifacts"] = None
            st.session_state["agent_report_text"] = None
            st.session_state["loaded_history_run_id"] = None
            st.session_state["active_run_id"] = None
            st.session_state["synced_run_id"] = None
            st.session_state["run_mode_selection"] = tr("preview_mode")
            st.session_state["preview_rows_selection"] = None
            st.session_state["build_hierarchy_selection"] = True

    df = st.session_state["uploaded_df"]
    if df is None:
        st.error(tr("parse_failed"))
        return

    column_left, column_right = st.columns([1, 1])
    with column_left:
        query_col = st.selectbox(
            tr("query_column"),
            options=list(df.columns),
            index=list(df.columns).index(st.session_state["query_col"]) if st.session_state["query_col"] in df.columns else 0,
        )
        st.session_state["query_col"] = query_col
    with column_right:
        st.session_state["display_name"] = st.text_input(tr("project_name"), value=st.session_state["display_name"])

    pricing = resolve_pricing_config(st.session_state["config"])
    preflight = preflight_service.analyze(df, query_col=query_col, pricing=pricing)
    st.session_state["preflight"] = preflight

    metric_cols = st.columns(6)
    metric_cols[0].metric(tr("rows"), f"{preflight['total_rows']:,}")
    metric_cols[1].metric(tr("valid_queries"), f"{preflight['valid_rows']:,}")
    metric_cols[2].metric(tr("empty_rows"), f"{preflight['empty_rows']:,}")
    metric_cols[3].metric(tr("duplicate_rows"), f"{preflight['duplicate_rows']:,}")
    metric_cols[4].metric(tr("embedding_tokens"), f"{preflight['estimated_embedding_tokens']:,}")
    metric_cols[5].metric(tr("suggested_preview"), f"{preflight['suggested_preview_rows']:,}")

    quality_flags = preflight.get("quality_flags", [])
    if quality_flags:
        st.warning("\n".join(f"- {translate_quality_flag(flag)}" for flag in quality_flags))

    cost_estimate = preflight["cost_estimate"]
    preview_estimate = cost_estimate["preview_run"]
    full_estimate = cost_estimate["full_run"]
    estimate_cols = st.columns(2)
    estimate_cols[0].write(f"**{tr('recommended_preview_estimate')}**")
    estimate_cols[0].write(tr("estimate_basis_rows", rows=f"{preview_estimate['row_count']:,}"))
    if preview_estimate.get("total_cost_rmb") is not None:
        estimate_cols[0].write(tr("estimate_total_cost", cost=format_rmb(preview_estimate["total_cost_rmb"])))
    estimate_cols[0].write(tr("input_tokens", tokens=f"{preview_estimate['embedding_tokens']:,}"))
    if preview_estimate.get("embedding_cost_rmb") is not None or preview_estimate.get("llm_cost_rmb") is not None:
        estimate_cols[0].write(
            tr(
                "estimate_breakdown",
                embedding=format_rmb(preview_estimate.get("embedding_cost_rmb")),
                llm=format_rmb(preview_estimate.get("llm_cost_rmb")),
            )
        )

    estimate_cols[1].write(f"**{tr('full_run_estimate')}**")
    estimate_cols[1].write(tr("estimate_basis_rows", rows=f"{full_estimate['row_count']:,}"))
    if full_estimate.get("total_cost_rmb") is not None:
        estimate_cols[1].write(tr("estimate_total_cost", cost=format_rmb(full_estimate["total_cost_rmb"])))
    estimate_cols[1].write(tr("input_tokens", tokens=f"{full_estimate['embedding_tokens']:,}"))
    if full_estimate.get("embedding_cost_rmb") is not None or full_estimate.get("llm_cost_rmb") is not None:
        estimate_cols[1].write(
            tr(
                "estimate_breakdown",
                embedding=format_rmb(full_estimate.get("embedding_cost_rmb")),
                llm=format_rmb(full_estimate.get("llm_cost_rmb")),
            )
        )

    st.markdown(f"**{tr('preview_data')}**")
    st.dataframe(df.head(20), use_container_width=True)


def render_run_tab():
    st.subheader(tr("run_title"))
    st.markdown(
        (
            '<div class="intentatlas-note">'
            + (
                "建议先跑 preview 做首轮验证。确认主题地图的形态和成本预估都符合预期后，再跑全量。"
                if current_language() == "zh"
                else "Preview is recommended for first-pass validation. Full runs are best after you are satisfied with the shape of the intent map and the cost estimate."
            )
            + "</div>"
        ),
        unsafe_allow_html=True,
    )
    if st.session_state["preflight"] is None or st.session_state["uploaded_df"] is None:
        st.info(tr("run_prereq"))
        return

    preflight = st.session_state["preflight"]
    config = st.session_state["config"]
    if preflight["valid_rows"] == 0:
        st.error(tr("no_valid_queries"))
        return

    active_run_id = st.session_state.get("active_run_id")
    active_snapshot = RUN_SESSION_SERVICE.get_session(active_run_id) if active_run_id else None
    if active_snapshot:
        if active_snapshot["status"] in {"running", "queued"}:
            st.info(tr("run_in_progress"))
        render_active_run_console(active_run_id)
        return

    preview_default = min(preflight["suggested_preview_rows"], max(1, preflight["valid_rows"]))
    preview_selection = st.session_state.get("preview_rows_selection")
    if preview_selection is None:
        st.session_state["preview_rows_selection"] = preview_default
    else:
        st.session_state["preview_rows_selection"] = min(max(1, int(preview_selection)), max(1, preflight["valid_rows"]))

    mode_options = [tr("preview_mode"), tr("full_mode")]
    selected_mode = st.session_state.get("run_mode_selection", tr("preview_mode"))
    if selected_mode not in mode_options:
        selected_mode = tr("preview_mode")
        st.session_state["run_mode_selection"] = selected_mode
    mode = st.radio(
        tr("run_mode"),
        options=mode_options,
        horizontal=True,
        index=mode_options.index(selected_mode),
        key="run_mode_selection",
    )
    preview_rows = st.number_input(
        tr("preview_rows"),
        min_value=1,
        max_value=max(1, preflight["valid_rows"]),
        value=max(1, int(st.session_state["preview_rows_selection"])),
        step=1 if preflight["valid_rows"] < 50 else 50,
        disabled=mode != tr("preview_mode"),
        key="preview_rows_selection",
    )
    build_hierarchy = st.checkbox(
        tr("build_hierarchy"),
        value=bool(st.session_state.get("build_hierarchy_selection", True)),
        key="build_hierarchy_selection",
    )

    if not is_model_configured(config):
        st.error(tr("config_required"))
        return

    if st.button(tr("run_button"), type="primary", use_container_width=True):
        apply_runtime_config(config)
        analysis_service = AnalysisService()
        normalized_df = normalize_for_run(st.session_state["uploaded_df"], st.session_state["query_col"])
        display_name = st.session_state["display_name"].strip() or Path(st.session_state["uploaded_name"]).stem
        internal_dataset_name = build_internal_dataset_name(
            display_name=display_name,
            file_bytes=st.session_state["uploaded_bytes"] or b"",
            mode=mode,
            preview_rows=int(preview_rows) if mode == tr("preview_mode") else None,
        )

        if mode == tr("preview_mode"):
            run_df = normalized_df.sample(n=min(int(preview_rows), len(normalized_df)), random_state=42)
        else:
            run_df = normalized_df

        analysis_service.persist_dataframe_as_dataset(run_df, internal_dataset_name)
        run_id = RUN_SESSION_SERVICE.start_run(
            dataset_name=internal_dataset_name,
            display_name=display_name,
            run_mode=mode,
            sample_size=int(preview_rows) if mode == tr("preview_mode") else None,
            report_language=current_language(),
            build_hierarchy=build_hierarchy,
            enable_llm_report_summary=True,
            source_df=st.session_state["uploaded_df"],
            uploaded_name=st.session_state.get("uploaded_name"),
            query_column=st.session_state.get("query_col") or "query",
            preflight=preflight,
            preview_rows=int(preview_rows) if mode == tr("preview_mode") else None,
        )
        st.session_state["active_run_id"] = run_id
        st.session_state["synced_run_id"] = None
        st.rerun()

    if active_run_id:
        render_active_run_console(active_run_id)


@st.fragment(run_every=1)
def render_active_run_console(run_id: str):
    snapshot = RUN_SESSION_SERVICE.get_session(run_id)
    if snapshot is None:
        st.session_state["active_run_id"] = None
        return

    sync_run_snapshot(snapshot)

    st.markdown(f"**{tr('active_run')}**")
    st.caption(tr("run_monitor_hint"))
    st.progress(int(snapshot["progress"]))

    control_cols = st.columns([1.4, 1, 1, 1])
    control_cols[0].metric(tr("run_status"), friendly_status_label(snapshot["status"]))
    control_cols[1].metric(tr("current_focus"), friendly_stage_label(snapshot["stage"]))

    metrics = snapshot["metrics"]
    if snapshot["stage"] == "embedding" and metrics.get("embedding_total", 0):
        control_cols[2].metric(
            tr("embedded"),
            f"{metrics.get('embedding_completed', 0):,}/{metrics.get('embedding_total', 0):,}",
        )
    else:
        control_cols[2].metric(tr("reviewed"), metrics.get("reviewed", 0))
    control_cols[3].metric(tr("themes"), metrics.get("categories", 0))

    manifest_record = RunHistoryService().load_manifest(run_id) or {}

    if snapshot["status"] in {"running", "queued"}:
        if st.button(tr("stop_run"), type="secondary", use_container_width=True):
            if RUN_SESSION_SERVICE.request_stop(run_id):
                st.warning(tr("run_cancel_requested"))
                st.rerun()
    else:
        action_cols = st.columns(2)
        if can_resume_record(manifest_record) and action_cols[0].button(tr("continue_run"), use_container_width=True):
            continue_saved_project(manifest_record)
        if action_cols[1].button(tr("edit_run_config"), use_container_width=True):
            st.session_state["active_run_id"] = None
            st.rerun()
        if can_resume_record(manifest_record):
            st.info(tr("run_can_resume"))

    secondary_cols = st.columns(4)
    secondary_cols[0].metric(tr("create"), metrics.get("create", 0))
    secondary_cols[1].metric(tr("merge"), metrics.get("merge", 0))
    secondary_cols[2].metric(tr("split"), metrics.get("split", 0))
    secondary_cols[3].metric(tr("inspect"), metrics.get("inspect", 0))

    tertiary_cols = st.columns(3)
    tertiary_cols[0].metric(tr("residual_groups"), metrics.get("residual", 0))
    tertiary_cols[1].metric(
        tr("remaining_work"),
        metrics.get("pending_tasks", 0) + metrics.get("pending_microclusters", 0),
    )
    tertiary_cols[2].metric(tr("valid_queries"), metrics.get("total_queries", 0))

    st.markdown(f"**{tr('recent_activity')}**")
    for item in reversed(snapshot.get("activity", [])[-8:]):
        st.markdown(f"- `{item['time']}` {item['message']}")

    if snapshot["status"] == "completed":
        artifacts = snapshot.get("artifacts") or {}
        st.success(
            tr(
                "run_success",
                run_mode=artifacts.get("run_mode", "unknown"),
                runtime_seconds=float(artifacts.get("runtime_seconds") or 0.0),
                path=artifacts.get("agent_report_path", "unknown"),
            )
        )
    elif snapshot["status"] == "cancelled":
        st.warning(tr("run_cancelled"))
    elif snapshot["status"] == "failed":
        st.error(tr("run_failed", error=snapshot.get("error", "unknown")))


def render_report_tab():
    st.subheader(tr("report_title"))
    st.markdown(
        (
            '<div class="intentatlas-note">'
            + (
                "这里是主要的产品结果层。默认交付物是一个单文件 <code>agent_report.md</code>，既适合人阅读，也适合继续丢给其他大模型追问。"
                if current_language() == "zh"
                else "This report surface is the main product output. The default handoff artifact is one <code>agent_report.md</code> file designed for both human reading and follow-up prompting in another LLM."
            )
            + "</div>"
        ),
        unsafe_allow_html=True,
    )
    artifacts = st.session_state.get("run_artifacts")
    if artifacts is None:
        st.info(tr("report_hint"))
        return

    summary = artifacts.summary if hasattr(artifacts, "summary") else artifacts.get("summary")
    hierarchy = artifacts.hierarchy if hasattr(artifacts, "hierarchy") else artifacts.get("hierarchy")
    agent_report_path = artifacts.agent_report_path if hasattr(artifacts, "agent_report_path") else Path(artifacts["agent_report_path"])
    agent_report_text = st.session_state.get("agent_report_text") or agent_report_path.read_text(encoding="utf-8")

    top_categories = ordered_categories(summary)
    residuals = ordered_residuals(summary)

    metric_cols = st.columns(5)
    metric_cols[0].metric(tr("leaf_categories"), summary.get("total_categories", 0))
    metric_cols[1].metric(tr("classified_queries"), summary.get("classified_query_count", summary.get("total_queries", 0)))
    metric_cols[2].metric(tr("residual_queries"), summary.get("residual_query_count", 0))
    metric_cols[3].metric(tr("run_mode"), artifact_value(artifacts, "run_mode", "loaded"))
    metric_cols[4].metric(tr("runtime_seconds_metric"), artifact_value(artifacts, "runtime_seconds", "n/a"))

    inner_tabs = st.tabs([tr("overview"), tr("categories_tab"), tr("hierarchy_tab"), tr("residuals_tab"), tr("agent_report_tab")])
    with inner_tabs[0]:
        if top_categories:
            chart_df = pd.DataFrame(
                {
                    "category": [item["name"][:48] for item in top_categories[:15]],
                    "query_count": [item["query_count"] for item in top_categories[:15]],
                }
            ).set_index("category")
            st.bar_chart(chart_df)
            st.write(tr("top_category_snapshot"))
            st.dataframe(
                pd.DataFrame(top_categories[:20])[["name", "query_count", "definition"]],
                use_container_width=True,
            )
        else:
            st.info(tr("no_categories"))

    with inner_tabs[1]:
        for category in top_categories:
            with st.expander(f"{category['name']} ({category['query_count']})"):
                st.write(f"**{tr('definition')}**: {category['definition']}")
                st.write(f"**{tr('boundary_notes')}**: {category.get('boundary_notes') or 'none'}")
                examples = category.get("positive_examples", [])
                if examples:
                    st.write(f"**{tr('examples')}**")
                    for example in examples:
                        st.write(f"- {example}")

    with inner_tabs[2]:
        if hierarchy and hierarchy.get("roots"):
            st.markdown(render_hierarchy_markdown(hierarchy["roots"], summary.get("categories", {})))
        else:
            st.info(tr("no_hierarchy"))

    with inner_tabs[3]:
        if residuals:
            for index, cluster in enumerate(residuals, 1):
                with st.expander(f"{tr('residual_groups')} {index} ({cluster['query_count']})"):
                    st.write(f"**{tr('reason')}**: {cluster['reason']}")
                    samples = cluster.get("sample_texts", [])
                    for sample in samples:
                        st.write(f"- {sample}")
        else:
            st.success(tr("no_residuals"))

    with inner_tabs[4]:
        action_cols = st.columns(2)
        if action_cols[0].button(
            tr("regenerate_ai_report"),
            help=tr("regenerate_ai_report_help"),
            use_container_width=True,
        ):
            if not is_llm_configured(st.session_state["config"]):
                st.warning(tr("regenerate_ai_report_requires_model"))
            else:
                regenerate_current_agent_report()
                agent_report_text = st.session_state.get("agent_report_text") or agent_report_text
                st.success(tr("regenerate_ai_report_success"))

        action_cols[1].download_button(
            tr("download_agent_report"),
            data=agent_report_text,
            file_name=agent_report_path.name,
            mime="text/markdown",
            use_container_width=True,
        )
        preview_body, appendix_body = agent_report_text, ""
        appendix_marker = "## Structured Appendix"
        if appendix_marker not in agent_report_text:
            appendix_marker = "## 结构化附录"
        if appendix_marker in agent_report_text:
            preview_body, appendix_body = agent_report_text.split(appendix_marker, 1)
        with st.expander(tr("preview_agent_report"), expanded=True):
            st.markdown(preview_body)
        if appendix_body:
            with st.expander(tr("structured_appendix_preview"), expanded=False):
                st.markdown(tr("structured_appendix_heading"))
                st.markdown(appendix_body)
        with st.expander(tr("raw_artifacts"), expanded=False):
            summary_path = artifact_value(artifacts, "summary_path")
            hierarchy_path = artifact_value(artifacts, "hierarchy_path")
            csv_path = artifact_value(artifacts, "csv_path")
            if summary_path:
                st.code(str(summary_path))
            if hierarchy_path:
                st.code(str(hierarchy_path))
            if csv_path:
                st.code(str(csv_path))


def normalize_for_run(df: pd.DataFrame, query_col: str) -> pd.DataFrame:
    normalized = df.copy()
    if query_col != "query":
        normalized = normalized.rename(columns={query_col: "query"})
    normalized["query"] = normalized["query"].fillna("").astype(str).str.strip()
    normalized = normalized[normalized["query"] != ""].reset_index(drop=True)
    return normalized


def build_internal_dataset_name(
    *,
    display_name: str,
    file_bytes: bytes,
    mode: str,
    preview_rows: Optional[int],
) -> str:
    safe_name = AnalysisService().sanitize_dataset_name(display_name)
    fingerprint_source = file_bytes + f"|{mode}|{preview_rows}".encode("utf-8")
    digest = hashlib.sha1(fingerprint_source).hexdigest()[:8]
    if mode == "preview":
        return f"{safe_name}_preview_{preview_rows}_{digest}"
    return f"{safe_name}_full_{digest}"


def stable_model_key(value: str) -> str:
    normalized = value.strip() or "unknown"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]


def format_rmb(value: Optional[float]) -> str:
    if value is None:
        return "0"
    if value >= 1:
        return f"{value:.2f}"
    if value >= 0.01:
        return f"{value:.4f}"
    return f"{value:.6f}"


def is_model_configured(config: Dict[str, Any]) -> bool:
    llm = config.get("llm", {})
    embedding = config.get("embedding", {})
    placeholders = {"", "your-llm-api-key-here", "your-embedding-api-key-here"}
    return (
        llm.get("api_key") not in placeholders
        and embedding.get("api_key") not in placeholders
        and bool(llm.get("model_name"))
        and bool(embedding.get("model_name"))
    )


def is_llm_configured(config: Dict[str, Any]) -> bool:
    llm = config.get("llm", {})
    placeholders = {"", "your-llm-api-key-here"}
    return llm.get("api_key") not in placeholders and bool(llm.get("model_name"))


def ordered_categories(summary: Dict[str, Any]) -> list[Dict[str, Any]]:
    categories = []
    for category_id, category in summary.get("categories", {}).items():
        categories.append({"id": category_id, **category})
    categories.sort(key=lambda item: (-item["query_count"], item["id"]))
    return categories


def ordered_residuals(summary: Dict[str, Any]) -> list[Dict[str, Any]]:
    residuals = []
    for cluster_id, cluster in summary.get("residual_clusters", {}).items():
        residuals.append({"id": cluster_id, **cluster})
    residuals.sort(key=lambda item: (-item["query_count"], item["id"]))
    return residuals


def artifact_value(artifacts: Any, key: str, default: Any = None) -> Any:
    if hasattr(artifacts, key):
        return getattr(artifacts, key)
    if isinstance(artifacts, dict):
        return artifacts.get(key, default)
    return default


def sync_run_snapshot(snapshot: Dict[str, Any]):
    run_id = snapshot.get("run_id")
    if snapshot.get("status") != "completed" or st.session_state.get("synced_run_id") == run_id:
        return

    artifacts = snapshot.get("artifacts") or {}
    if not artifacts:
        return
    if "summary" not in artifacts and st.session_state.get("run_artifacts") is not None:
        st.session_state["synced_run_id"] = run_id
        return

    agent_report_path = artifacts.get("agent_report_path")
    if agent_report_path:
        st.session_state["agent_report_text"] = Path(agent_report_path).read_text(encoding="utf-8")
    st.session_state["run_artifacts"] = artifacts
    st.session_state["synced_run_id"] = run_id


def friendly_status_label(status: str) -> str:
    return {
        "running": tr("status_running"),
        "completed": tr("status_completed"),
        "cancelled": tr("status_cancelled"),
        "failed": tr("status_failed"),
        "queued": tr("status_queued"),
    }.get(status, status)


def friendly_stage_label(stage: str) -> str:
    return {
        "loading": tr("stage_loading"),
        "embedding": tr("stage_embedding"),
        "taxonomy": tr("stage_taxonomy"),
        "saving": tr("stage_saving"),
        "hierarchy": tr("stage_hierarchy"),
        "report": tr("stage_report"),
        "completed": tr("status_completed"),
        "cancelled": tr("status_cancelled"),
        "failed": tr("status_failed"),
        "queued": tr("status_queued"),
    }.get(stage, stage)


def render_hierarchy_markdown(
    roots: list[Dict[str, Any]],
    category_map: Optional[Dict[str, Any]] = None,
    depth: int = 0,
) -> str:
    lines = []
    prefix = "  " * depth
    category_map = category_map or {}
    for node in roots:
        if "leaf_id" in node:
            leaf_id = node["leaf_id"]
            leaf_name = category_map.get(leaf_id, {}).get("name", leaf_id)
            lines.append(f"{prefix}- {leaf_name}")
        else:
            lines.append(f"{prefix}- **{node['name']}**: {node['description']}")
            lines.append(render_hierarchy_markdown(node.get("children", []), category_map, depth + 1))
    return "\n".join(line for line in lines if line)


def format_saved_run_label(record: Dict[str, Any]) -> str:
    timestamp = format_timestamp_label(record.get("timestamp", "unknown"))
    display_name = record.get("display_name") or record.get("dataset_name") or "unknown"
    run_mode = record.get("run_mode", "unknown")
    source = record.get("source")
    status = friendly_status_label(record.get("status", "completed"))
    if source == "legacy":
        return f"{display_name} · {timestamp} · {run_mode} · {status} · {tr('saved_run_legacy')}"
    return f"{display_name} · {timestamp} · {run_mode} · {status}"


def format_timestamp_label(value: str) -> str:
    if len(value) == 15 and "_" in value:
        return f"{value[:4]}-{value[4:6]}-{value[6:8]} {value[9:11]}:{value[11:13]}:{value[13:15]}"
    return value


def load_saved_project(record: Dict[str, Any]):
    history_service = RunHistoryService()
    bundle = history_service.load_project_bundle(record, report_language=current_language())
    source_df = bundle.get("source_df")
    if source_df is not None:
        st.session_state["uploaded_df"] = source_df
        st.session_state["query_col"] = bundle.get("query_column", "query") if bundle.get("query_column") in source_df.columns else source_df.columns[0]
        st.session_state["display_name"] = record.get("display_name") or record.get("dataset_name")
        st.session_state["uploaded_name"] = record.get("uploaded_name") or f"{st.session_state['display_name']}.csv"
        st.session_state["uploaded_bytes"] = source_df.to_csv(index=False).encode("utf-8")
        st.session_state["preflight"] = bundle.get("preflight")
    elif record.get("source") == "legacy":
        st.session_state["uploaded_df"] = None
        st.session_state["preflight"] = None
        st.session_state["uploaded_name"] = None
        st.session_state["uploaded_bytes"] = None

    run_artifacts = bundle.get("run_artifacts")
    if run_artifacts is not None:
        st.session_state["run_artifacts"] = run_artifacts
        st.session_state["agent_report_text"] = run_artifacts["agent_report_text"]
    else:
        st.session_state["run_artifacts"] = None
        st.session_state["agent_report_text"] = None

    preflight = bundle.get("preflight") or {}
    valid_rows = int(preflight.get("valid_rows", 0) or 0)
    preview_default = min(valid_rows, 512) if valid_rows else None
    st.session_state["run_mode_selection"] = record.get("run_mode") or tr("preview_mode")
    st.session_state["preview_rows_selection"] = min(
        max(1, int(record.get("preview_rows") or preview_default or 1)),
        max(1, valid_rows or int(record.get("preview_rows") or 1)),
    )
    st.session_state["build_hierarchy_selection"] = bool(record.get("build_hierarchy", True))

    session_snapshot = bundle.get("session_snapshot") or build_session_snapshot_fallback(record, run_artifacts)
    if record.get("status") == "running" and RUN_SESSION_SERVICE.get_session(record.get("run_id")) is None:
        if session_snapshot:
            session_snapshot = session_snapshot.copy()
            session_snapshot["status"] = "cancelled"
            session_snapshot["stage"] = "cancelled"
            activity = list(session_snapshot.get("activity", []))
            activity.append(
                {
                    "kind": "warning",
                    "message": "检测到上次运行没有正常结束，可以继续运行。",
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
            )
            session_snapshot["activity"] = activity
        record["status"] = "cancelled"
        history_service.upsert_manifest(record)

    if session_snapshot:
        RUN_SESSION_SERVICE.register_persisted_session(
            record["run_id"],
            session_snapshot,
            dataset_name=record.get("dataset_name"),
            display_name=record.get("display_name"),
            run_mode=record.get("run_mode", "unknown"),
        )
        st.session_state["active_run_id"] = record.get("run_id")
    else:
        st.session_state["active_run_id"] = None

    st.session_state["loaded_history_run_id"] = record.get("run_id")
    st.session_state["synced_run_id"] = None
    st.sidebar.success(tr("project_restored", label=format_saved_run_label(record)))


def can_resume_record(record: Dict[str, Any]) -> bool:
    if not record:
        return False
    if record.get("source") == "legacy":
        return False
    return record.get("status") in {"cancelled", "failed", "running"} and bool(record.get("dataset_name"))


def build_session_snapshot_fallback(
    record: Dict[str, Any],
    run_artifacts: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not record:
        return None

    status = record.get("status", "completed")
    if status not in {"completed", "cancelled", "failed", "running"}:
        return None

    summary = (run_artifacts or {}).get("summary") or {}
    total_queries = (
        summary.get("total_queries")
        or summary.get("classified_query_count", 0) + summary.get("residual_query_count", 0)
        or record.get("source_row_count")
        or 0
    )
    metrics = {
        "reviewed": summary.get("total_categories", 0),
        "create": 0,
        "merge": 0,
        "split": 0,
        "inspect": 0,
        "categories": summary.get("total_categories", 0),
        "residual": summary.get("residual_cluster_count", len(summary.get("residual_clusters", {}))),
        "pending_tasks": 0,
        "pending_microclusters": 0,
        "embedding_completed": total_queries if status == "completed" else 0,
        "embedding_total": total_queries,
        "total_queries": total_queries,
    }
    activity = [
        {
            "kind": "info",
            "message": (
                "已从保存的项目记录中恢复运行上下文。"
                if current_language() == "zh"
                else "Run context restored from saved project history."
            ),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    ]
    return {
        "run_id": record.get("run_id"),
        "dataset_name": record.get("dataset_name"),
        "display_name": record.get("display_name") or record.get("dataset_name"),
        "run_mode": record.get("run_mode", "full"),
        "status": status,
        "stage": "completed" if status == "completed" else status,
        "progress": 100 if status == "completed" else 64,
        "created_at": record.get("timestamp"),
        "finished_at": record.get("timestamp"),
        "stop_requested": status == "cancelled",
        "error": None,
        "artifacts": {
            "dataset_name": record.get("dataset_name"),
            "display_name": record.get("display_name") or record.get("dataset_name"),
            "run_mode": record.get("run_mode", "full"),
            "csv_path": str(run_artifacts.get("csv_path")) if run_artifacts and run_artifacts.get("csv_path") else None,
            "summary_path": str(run_artifacts.get("summary_path")) if run_artifacts and run_artifacts.get("summary_path") else record.get("summary_path"),
            "hierarchy_path": str(run_artifacts.get("hierarchy_path")) if run_artifacts and run_artifacts.get("hierarchy_path") else record.get("hierarchy_path"),
            "agent_report_path": str(run_artifacts.get("agent_report_path")) if run_artifacts and run_artifacts.get("agent_report_path") else record.get("agent_report_path"),
            "manifest_path": record.get("manifest_path"),
            "runtime_seconds": record.get("runtime_seconds"),
        }
        if run_artifacts or record.get("summary_path")
        else None,
        "activity": activity,
        "metrics": metrics,
    }


def continue_saved_project(record: Dict[str, Any]):
    apply_runtime_config(st.session_state["config"])
    run_id = RUN_SESSION_SERVICE.start_run(
        run_id=record.get("run_id"),
        dataset_name=record.get("dataset_name"),
        display_name=record.get("display_name") or record.get("dataset_name"),
        run_mode=record.get("run_mode", tr("full_mode")),
        sample_size=record.get("preview_rows") if record.get("run_mode") == tr("preview_mode") else None,
        report_language=record.get("report_language") or current_language(),
        build_hierarchy=bool(record.get("build_hierarchy", True)),
        enable_llm_report_summary=True,
        source_data_path=record.get("source_data_path"),
        source_row_count=record.get("source_row_count"),
        uploaded_name=record.get("uploaded_name"),
        query_column=record.get("query_column") or "query",
        preflight=record.get("preflight"),
        preview_rows=record.get("preview_rows"),
        resume_state_path=record.get("resume_state_path"),
    )
    st.session_state["active_run_id"] = run_id
    st.session_state["loaded_history_run_id"] = run_id
    st.session_state["synced_run_id"] = None
    st.rerun()


def regenerate_current_agent_report():
    artifacts = st.session_state.get("run_artifacts")
    if artifacts is None:
        return

    apply_runtime_config(st.session_state["config"])
    report_service = ReportService()
    llm_service = LLMService(st.session_state["config"].get("llm", {}))

    summary = artifact_value(artifacts, "summary", {})
    hierarchy = artifact_value(artifacts, "hierarchy")
    agent_report_path = Path(artifact_value(artifacts, "agent_report_path"))
    report_text = report_service.build_agent_report(
        summary=summary,
        hierarchy=hierarchy,
        language=current_language(),
        llm_service=llm_service,
        enable_llm_summary=True,
        run_metadata={
            "run_mode": artifact_value(artifacts, "run_mode", "loaded"),
            "display_name": artifact_value(artifacts, "display_name", summary.get("display_name")),
            "runtime_seconds": artifact_value(artifacts, "runtime_seconds", "cached"),
            "llm_model": st.session_state["config"].get("llm", {}).get("model_name", "unknown"),
            "embedding_model": st.session_state["config"].get("embedding", {}).get("model_name", "unknown"),
        },
    )
    agent_report_path.write_text(report_text, encoding="utf-8")
    st.session_state["agent_report_text"] = report_text
    if isinstance(artifacts, dict):
        artifacts["agent_report_path"] = agent_report_path


if __name__ == "__main__":
    main()
