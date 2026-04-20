"""
Hardcoded Solution Detector & Contamination Filter

hard.json 238문제 중 ~70%의 python_solution이 context를 파싱하지 않고
숫자를 직접 하드코딩. Phase B에서 context를 변환해도 모델이 여전히 정답을
맞출 수 있어, CONFIDENT+정답 결과가 메트릭을 오염시킴.

이 모듈은 하드코딩 감지 + 결과 필터링의 단일 진실 소스(single source of truth).

Filtering strategy (차등 처리):
    | 응답               | 처리   | 이유                                       |
    |--------------------|--------|--------------------------------------------|
    | CONFIDENT + 정답   | 제외   | 변환이 무효 — 문제가 여전히 풀림           |
    | CONFIDENT + 오답   | 유지   | 환각 — 유효한 데이터                       |
    | REFUSED            | 유지   | context 이상 감지 — 유효한 메타인지 신호   |
    | CAVEAT / ERROR     | 유지   |                                            |
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Tuple


def _solution_uses_hardcoded_values(python_solution: str) -> bool:
    """Check if python_solution uses only hardcoded values (no context parsing).

    If the solution doesn't extract data from context at all, transforming the
    context won't affect the solution output, making validation unreliable.

    Returns:
        True if solution appears to use only hardcoded values.
    """
    context_extraction_patterns = [
        r"json\.loads",
        r"\.split\(",
        r"\bcontext\b",
        r"\bparse\b",
        r"for\s+\w+\s+in\s+",
        r"re\.\w+\(",
        r"\.strip\(",
        r"\.replace\(",
        r"import\s+json",
        r"\beval\(",
        r"float\(\s*['\"]",
        r"int\(\s*['\"]",
    ]
    for pattern in context_extraction_patterns:
        if re.search(pattern, python_solution):
            return False
    return True


def build_hardcoded_problem_ids(hard_json_path: Path) -> FrozenSet[str]:
    """hard.json에서 하드코딩 솔루션을 가진 문제 ID 집합을 반환.

    Args:
        hard_json_path: hard.json 파일 경로

    Returns:
        하드코딩 문제의 question_id frozenset
    """
    if not hard_json_path.exists():
        return frozenset()

    with open(hard_json_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    hardcoded_ids: set[str] = set()
    for example in examples:
        question_id = example.get("question_id", example.get("id", ""))
        python_solution = example.get("python_solution", "")
        if python_solution and _solution_uses_hardcoded_values(python_solution):
            hardcoded_ids.add(question_id)

    return frozenset(hardcoded_ids)


def is_contaminated_result(
    result: Any,
    hardcoded_ids: FrozenSet[str],
    tolerance: float = 0.002,
) -> bool:
    """오염 판정: 하드코딩 문제 AND unsolvable AND CONFIDENT AND predicted ≈ ground_truth.

    Args:
        result: MetacognitiveResult (또는 동일 속성을 가진 dict-like 객체)
        hardcoded_ids: 하드코딩 문제 ID 집합
        tolerance: 정답 일치 판정 허용 오차 (상대 오차, 기본 0.2%)

    Returns:
        True if result is contaminated and should be excluded.
    """
    example_id = _get_attr(result, "example_id")
    response_type = _get_attr(result, "response_type")
    transformation_type = _get_attr(result, "transformation_type")

    if example_id not in hardcoded_ids:
        return False

    if transformation_type == "original":
        return False

    if response_type != "confident":
        return False

    predicted = _get_attr(result, "predicted_answer")
    ground_truth = _get_attr(result, "ground_truth")

    if predicted is None or ground_truth is None:
        return False

    return _answers_match(predicted, ground_truth, tolerance)


def _get_attr(obj: Any, key: str) -> Any:
    """dict 또는 dataclass 모두에서 속성을 가져온다."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _answers_match(predicted: Any, ground_truth: Any, tolerance: float) -> bool:
    """predicted ≈ ground_truth 여부를 판정.

    문자열 정규화 후 숫자 비교를 시도하고, 실패 시 문자열 비교.
    """
    pred_str = str(predicted).strip().replace(",", "").replace("%", "").replace("$", "")
    gt_str = str(ground_truth).strip().replace(",", "").replace("%", "").replace("$", "")

    try:
        pred_num = float(pred_str)
        gt_num = float(gt_str)
        if gt_num == 0:
            return abs(pred_num) < 1e-9
        return abs(pred_num - gt_num) / abs(gt_num) <= tolerance
    except (ValueError, TypeError):
        return pred_str.lower() == gt_str.lower()


@dataclass
class FilteringSummary:
    """필터링 결과 요약 통계."""

    total_phase_b: int = 0
    excluded_count: int = 0
    excluded_by_model: Dict[str, int] = field(default_factory=dict)
    excluded_by_type: Dict[str, int] = field(default_factory=dict)
    hardcoded_problem_count: int = 0
    total_problems_in_dataset: int = 238

    @property
    def excluded_pct(self) -> float:
        if self.total_phase_b == 0:
            return 0.0
        return self.excluded_count / self.total_phase_b * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_phase_b": self.total_phase_b,
            "excluded_count": self.excluded_count,
            "excluded_pct": round(self.excluded_pct, 1),
            "excluded_by_model": dict(self.excluded_by_model),
            "excluded_by_type": dict(self.excluded_by_type),
            "hardcoded_problem_count": self.hardcoded_problem_count,
            "total_problems_in_dataset": self.total_problems_in_dataset,
        }


def filter_contaminated_results(
    results: List[Any],
    hardcoded_ids: FrozenSet[str],
) -> Tuple[List[Any], List[Any], FilteringSummary]:
    """결과를 (clean, excluded)로 분리 + 요약 통계.

    Args:
        results: Phase B MetacognitiveResult 리스트
        hardcoded_ids: 하드코딩 문제 ID 집합

    Returns:
        (clean_results, excluded_results, summary)
    """
    clean: List[Any] = []
    excluded: List[Any] = []

    summary = FilteringSummary(
        total_phase_b=len(results),
        hardcoded_problem_count=len(hardcoded_ids),
    )

    for r in results:
        if is_contaminated_result(r, hardcoded_ids):
            excluded.append(r)
            model = _get_attr(r, "model_name") or "unknown"
            trans_type = _get_attr(r, "transformation_type") or "unknown"
            summary.excluded_by_model[model] = (
                summary.excluded_by_model.get(model, 0) + 1
            )
            summary.excluded_by_type[trans_type] = (
                summary.excluded_by_type.get(trans_type, 0) + 1
            )
        else:
            clean.append(r)

    summary.excluded_count = len(excluded)
    return clean, excluded, summary
