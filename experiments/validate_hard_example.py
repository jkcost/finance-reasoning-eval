"""Hard Example Answer Validator.

extract_ic_l1_l2_candidates.py 가 만든 candidates JSON을 입력으로 받아,
각 candidate에 대해 원본/수정된 Python solution을 **격리된 sandbox에서 실행**하고
새 ground truth를 계산한다.

안전장치:
    - AST 레벨 import 검사: 표준 수학 라이브러리만 허용
    - 실행 시간 제한 (기본 5초)
    - stdout 캡처 → 최종 수치 추출
    - 예외/타임아웃 시 result["ok"] = False

**중요**: 이 스크립트는 "candidate가 실제로 치환으로 새 답이 나오는가?"를 자동 체크한다.
"어느 값을 치환해야 하는가"는 LLM의 removed_or_modified 메타정보 + matched_literal 힌트 기반.
최종 채택 여부는 사람 검수(generate_hard_example_review.py) 단계에서 결정.

출력:
    experiments/results/metacognitive/hard_example_validated.json
        {
          "metadata": {...},
          "validated": [
            {
              "question_id": ...,
              "ic_level": ...,
              "original_answer": "100.0",
              "new_answer": "1000.0",
              "delta_ratio": 10.0,
              "replacement": {"from": 500000000, "to": 5000000000},
              "ok": true,
              "sandbox_log": "..."
            },
            ...
          ]
        }

Usage:
    python experiments/validate_hard_example.py \\
        --candidates experiments/results/metacognitive/hard_example_candidates.json \\
        --transformations experiments/results/metacognitive/batch_transformations_0_238.json
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


ALLOWED_IMPORTS = frozenset({"math", "statistics", "fractions", "decimal", "numpy", "pandas"})
BANNED_NAMES = frozenset({
    "exec", "eval", "compile", "__import__", "open",
    "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir",
})
BANNED_DUNDER = frozenset({
    "__class__", "__bases__", "__subclasses__", "__globals__",
    "__builtins__", "__dict__", "__getattribute__", "__mro__",
})
BANNED_ATTRIBUTE_CHAINS = frozenset({
    "os.system", "os.popen", "os.execv", "os.execve", "os.spawn",
    "subprocess.run", "subprocess.Popen", "subprocess.call",
    "socket.socket", "socket.create_connection",
    "pandas.read_csv", "pandas.read_json", "pandas.read_pickle", "pandas.read_parquet",
    "pandas.to_pickle", "pandas.to_csv",
    "numpy.load", "numpy.save", "numpy.fromfile", "numpy.tofile",
})

# Attribute leaf names that enable I/O or RCE regardless of alias.
# ``import numpy as np; np.load(...)`` bypasses chain match but not this leaf set.
BANNED_ATTRIBUTE_LEAVES = frozenset({
    "system", "popen", "execv", "execve", "spawn", "spawnl", "spawnlp",
    "Popen", "call", "check_output", "check_call",
    "socket", "create_connection", "urlopen",
    "read_csv", "read_json", "read_pickle", "read_parquet", "read_hdf",
    "read_excel", "read_sql", "read_table", "read_feather",
    "to_pickle", "to_csv", "to_json", "to_excel", "to_parquet", "to_feather",
    "load", "save", "fromfile", "tofile", "loadtxt", "savetxt",
    "genfromtxt", "load_npz",
})


def _resolve_attribute_chain(node: ast.AST) -> str | None:
    """Resolve ``os.system`` style attribute chains to dotted strings."""
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.insert(0, current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.insert(0, current.id)
        return ".".join(parts)
    return None


def _check_solution_safety(source: str) -> tuple[bool, str]:
    """Run AST-level safety check on a python solution.

    Blocks:
      - disallowed module imports (whitelist: math, statistics, fractions, decimal, numpy, pandas)
      - direct name calls like exec/eval/__import__/open/getattr/vars/...
      - dunder attribute access (__class__, __subclasses__, __globals__, ...)
      - attribute-chain calls like os.system, subprocess.run, socket.socket,
        pandas.read_csv / to_pickle, numpy.load
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return False, f"disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or "").split(".")[0]
            if base not in ALLOWED_IMPORTS:
                return False, f"disallowed from-import: {node.module}"
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_DUNDER:
                return False, f"disallowed dunder access: {node.attr}"
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                return False, f"disallowed name: {node.id}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
                return False, f"disallowed call: {node.func.id}"
            if isinstance(node.func, ast.Attribute):
                chain = _resolve_attribute_chain(node.func)
                if chain in BANNED_ATTRIBUTE_CHAINS:
                    return False, f"disallowed attribute call: {chain}"
                if node.func.attr in BANNED_ATTRIBUTE_LEAVES:
                    return False, f"disallowed attribute call (leaf): {node.func.attr}"
    return True, "ok"


def _wrap_solution_for_execution(source: str) -> str:
    """Ensure solution output is printed to stdout.

    FinanceReasoning solutions are typically ``def solution(): ... return value``.
    Without invocation, executing the source only defines the function. This wrapper
    appends ``print(solution())`` when a top-level ``solution`` function exists and
    no top-level ``print`` statement already appears.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    has_solution_fn = any(
        isinstance(node, ast.FunctionDef) and node.name == "solution"
        for node in tree.body
    )
    has_top_level_print = any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "print"
        for node in tree.body
    )

    if has_solution_fn and not has_top_level_print:
        return source.rstrip() + "\n\nprint(solution())\n"
    return source


def run_sandbox(source: str, timeout: float = 5.0) -> dict[str, Any]:
    """Execute source in a subprocess with timeout.

    Safety is checked via AST before invocation. The source is written to a temp file
    and run as ``python <tmp>`` so the subprocess is fully isolated from our process
    (works identically on fork/spawn platforms).

    Returns dict with keys: ``ok`` (bool), ``stdout``, ``error``.
    """
    ok, msg = _check_solution_safety(source)
    if not ok:
        return {"ok": False, "error": f"safety_check_failed: {msg}"}

    wrapped = _wrap_solution_for_execution(source)

    # Isolate the subprocess environment: strip API keys and other secrets so a
    # malicious solution cannot exfiltrate them via stdout. Only keep PATH and a
    # minimal locale so numpy/pandas imports still work.
    sanitized_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
        "HOME": tempfile.gettempdir(),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "solution.py"
        tmp_path.write_text(wrapped, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=sanitized_env,
                cwd=tmp_dir,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout_after_{timeout}s"}

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        last_line = stderr[-1] if stderr else "unknown_error"
        return {"ok": False, "error": last_line[:200]}

    return {"ok": True, "stdout": result.stdout or ""}


def _extract_last_number(stdout: str) -> str | None:
    """Parse the final numeric output line (supports floats, ints, scientific notation).

    Rejects NaN/Inf sentinels so that downstream delta comparison never misinterprets
    ``new_answer != original_answer`` as a successful transformation when both are NaN.
    """
    if not stdout:
        return None
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        tokens = line.replace(",", "").split()
        for token in reversed(tokens):
            try:
                value = float(token)
            except ValueError:
                continue
            if not math.isfinite(value):
                return None
            return token
    return None


def _replace_literal(source: str, old_value: float, new_value: float) -> tuple[str | None, int]:
    """Replace a numeric literal in source via AST → source transform.

    Returns (new_source, replacement_count). If old_value isn't found, returns (None, 0).
    Explicitly excludes bool constants (True/False) since ``isinstance(True, int)`` is True
    and matching float(True)==1.0 would corrupt control flow when old_value=1.0.
    Uses math.isclose with a small absolute tolerance to avoid IEEE-754 equality pitfalls.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, 0

    class Replacer(ast.NodeTransformer):
        def __init__(self) -> None:
            self.count = 0

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, bool):  # bool is int subclass; guard first
                return node
            if isinstance(node.value, (int, float)) and math.isclose(
                float(node.value), float(old_value), rel_tol=1e-9, abs_tol=1e-9
            ):
                self.count += 1
                return ast.copy_location(ast.Constant(value=new_value), node)
            return node

    replacer = Replacer()
    new_tree = replacer.visit(tree)
    if replacer.count == 0:
        return None, 0
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree), replacer.count
    except AttributeError:
        # Python < 3.9 fallback: unsupported
        return None, 0


@dataclass
class ValidationRecord:
    """Per-candidate validation outcome."""

    question_id: str
    ic_level: str
    original_answer: str | None
    new_answer: str | None
    matched_literal: float | None
    proposed_new_value: float | None
    replacement_count: int
    ok: bool
    reason: str


def _find_problem(batch: dict[str, Any], qid: str) -> dict[str, Any] | None:
    """Locate a problem record in the transformations batch by question_id."""
    for problem in batch.get("problems", []):
        if problem.get("question_id") == qid:
            return problem
    return None


def _proposed_new_value(ic_level: str, matched_literal: float) -> float:
    """Compute the proposed replacement value heuristically based on IC level.

    This is a first-pass heuristic; human review decides final adoption.
    """
    if ic_level == "IC-L1":
        return matched_literal * 10
    # IC-L2: 1.5x is the most common multiplier in the transformation spec
    return matched_literal * 1.5


def validate_candidate(
    candidate: dict[str, Any], batch: dict[str, Any]
) -> ValidationRecord:
    """Validate a single candidate by computing original & new answers."""
    qid = candidate["question_id"]
    ic_level = candidate["ic_level"]
    matched = candidate.get("matched_literal")

    problem = _find_problem(batch, qid)
    if problem is None:
        return ValidationRecord(
            qid, ic_level, None, None, matched, None, 0, False, "problem_not_found"
        )

    source = problem.get("python_solution", "")
    if not source.strip():
        return ValidationRecord(
            qid, ic_level, None, None, matched, None, 0, False, "empty_solution"
        )

    original_result = run_sandbox(source)
    if not original_result["ok"]:
        return ValidationRecord(
            qid, ic_level, None, None, matched, None, 0, False,
            f"original_failed: {original_result.get('error', '?')}",
        )
    original_answer = _extract_last_number(original_result.get("stdout", ""))

    if matched is None:
        return ValidationRecord(
            qid, ic_level, original_answer, None, None, None, 0, False, "no_matched_literal"
        )

    proposed_new = _proposed_new_value(ic_level, float(matched))
    new_source, count = _replace_literal(source, float(matched), proposed_new)
    if new_source is None or count == 0:
        return ValidationRecord(
            qid, ic_level, original_answer, None, matched, proposed_new, 0, False,
            "literal_replacement_missed",
        )

    new_result = run_sandbox(new_source)
    if not new_result["ok"]:
        return ValidationRecord(
            qid, ic_level, original_answer, None, matched, proposed_new, count, False,
            f"new_failed: {new_result.get('error', '?')}",
        )
    new_answer = _extract_last_number(new_result.get("stdout", ""))

    ok = new_answer is not None and original_answer != new_answer
    reason = "validated" if ok else "no_delta_or_unparseable"
    return ValidationRecord(
        qid, ic_level, original_answer, new_answer, matched, proposed_new, count, ok, reason,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Validate hard-example candidates by executing solutions")
    parser.add_argument("--candidates", type=Path, required=True, help="hard_example_candidates.json")
    parser.add_argument("--transformations", type=Path, required=True, help="batch_transformations JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/metacognitive/hard_example_validated.json"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N candidates")
    return parser.parse_args()


def main() -> int:
    """Run validation over all candidates."""
    args = parse_args()
    if not args.candidates.exists():
        logger.error(f"Candidates file not found: {args.candidates}")
        return 1
    if not args.transformations.exists():
        logger.error(f"Transformations file not found: {args.transformations}")
        return 1

    cand_data = json.loads(args.candidates.read_text(encoding="utf-8"))
    batch = json.loads(args.transformations.read_text(encoding="utf-8"))

    candidates = cand_data.get("candidates", [])
    if args.limit:
        candidates = candidates[: args.limit]
    logger.info(f"Validating {len(candidates)} candidates")

    results: list[ValidationRecord] = []
    for i, cand in enumerate(candidates):
        rec = validate_candidate(cand, batch)
        results.append(rec)
        status = "✓" if rec.ok else "✗"
        logger.info(f"  [{i + 1:3d}/{len(candidates)}] {status} {rec.question_id} {rec.ic_level} | {rec.reason}")

    ok_count = sum(1 for r in results if r.ok)
    logger.info(f"Validated: {ok_count}/{len(results)} successful")

    output = {
        "metadata": {
            "candidates_file": str(args.candidates),
            "transformations_file": str(args.transformations),
            "total": len(results),
            "successful": ok_count,
        },
        "validated": [asdict(r) for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
