"""Shared helpers for loading evaluation result JSON files.

Extract to kill duplication across extract_existing_reasons,
compute_memorization_score, fit_salience_regression.

Tolerates multiple file shapes:
    - Top-level list[dict]
    - Dict with 'results' list
    - Dict with 'evaluations' list
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_eval_records(path: Path) -> list[dict[str, Any]]:
    """Load evaluation records from a JSON file, tolerating nested shapes.

    Raises ValueError if the file exists but doesn't contain a recognizable list.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "evaluations"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unexpected JSON shape in {path}")


def solvable_qids(original_context_path: Path | None) -> set[str] | None:
    """Return question_ids correctly answered on the original context.

    Returns None when path is None so callers can short-circuit filtering.
    Always coerces question_id to str to avoid int/str mismatch drops.
    """
    if original_context_path is None:
        return None
    records = load_eval_records(original_context_path)
    return {
        str(r["question_id"])
        for r in records
        if r.get("is_correct") and r.get("question_id") is not None
    }


def qid_in_solvable(qid: Any, solvable: set[str] | None) -> bool:
    """Check whether qid belongs to a solvable set; True when no filter given."""
    if solvable is None:
        return True
    return str(qid) in solvable
