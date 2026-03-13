"""
Graph state checkpoint persistence for resumable runs.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict


class GraphStateService:
    def __init__(self, output_dir: str | Path = "output"):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "run_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, run_id: str) -> Path:
        return self.checkpoint_dir / f"{run_id}_graph_state.pkl"

    def save_state(self, run_id: str, state: Dict[str, Any]) -> Path:
        path = self.checkpoint_path(run_id)
        with open(path, "wb") as file:
            pickle.dump(state, file)
        return path

    def load_state(self, path: str | Path) -> Dict[str, Any]:
        with open(path, "rb") as file:
            return pickle.load(file)

    def clear_state(self, path: str | Path | None):
        if not path:
            return
        candidate = Path(path)
        if candidate.exists():
            candidate.unlink()
