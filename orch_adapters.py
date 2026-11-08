"""Parsers referenced from orch.yaml. See docs/orchestrator.md §10.2."""

from __future__ import annotations

import json
import numbers
from pathlib import Path

_SKIP_LMEVAL_KEYS = {"name", "alias", "sample_len"}


def parse_eval_results(output_dir: Path) -> dict[str, float]:
    """Read ``eval_results.json`` written by the tool-use and science evals."""
    path = Path(output_dir) / "eval_results.json"
    payload = json.loads(path.read_text())
    return {
        "accuracy": float(payload["accuracy"]),
        "num_correct": float(payload["num_correct"]),
        "num_total": float(payload["num_total"]),
    }


def parse_lmeval(output_dir: Path) -> dict[str, float]:
    """Flatten the newest ``results_*.json`` under ``output_dir``.

    ``metric,none`` becomes ``metric``. Other ``metric,filter`` keys are kept.
    ``sample_len`` is not a score.
    """
    root = Path(output_dir)
    files = list(root.rglob("results_*.json"))
    if not files:
        raise FileNotFoundError(f"no results_*.json under {root}")
    newest = max(files, key=lambda path: (path.stat().st_mtime, path.name))
    payload = json.loads(newest.read_text())
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{newest} has no results object")
    return _flatten_lmeval(results)


def _flatten_lmeval(results: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task, task_data in results.items():
        if not isinstance(task_data, dict):
            continue
        for key, value in task_data.items():
            if key in _SKIP_LMEVAL_KEYS:
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                continue
            metric = key[: -len(",none")] if key.endswith(",none") else key
            metrics[f"{task}/{metric}"] = float(value)
    return metrics
