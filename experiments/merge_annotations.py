"""
Annotation Merge Script

작업자별 JSON annotation 파일을 통합하고, 판정 불일치를 감지/리포트.

Usage:
    # 기본: annotations/ 폴더의 모든 JSON 머지
    python experiments/merge_annotations.py

    # 특정 파일 지정
    python experiments/merge_annotations.py --files review_김철수.json review_이영희.json

    # 출력 경로 지정
    python experiments/merge_annotations.py --output annotations/merged.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_annotation_file(path: Path) -> Dict[str, Any]:
    """Load a single annotation JSON file.

    Handles two formats:
      - Exported from HTML: {"assignee": ..., "annotations": {...}}
      - Raw annotations dict: {"test-2000": {"EA-full": {...}}}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" in data:
        assignee = data.get("assignee") or _extract_name_from_filename(path)
        return {"assignee": assignee, "annotations": data["annotations"]}

    # Raw format — infer assignee from filename
    return {"assignee": _extract_name_from_filename(path), "annotations": data}


def _extract_name_from_filename(path: Path) -> str:
    """Extract reviewer name from filename like review_김진수_0_238_2026-04-14.json."""
    stem = path.stem
    if stem.startswith("review_"):
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return stem


def merge_annotations(
    files: List[Path],
) -> Dict[str, Any]:
    """Merge multiple annotation files.

    Returns:
        {
            "merged_at": str,
            "sources": [{"assignee": str, "file": str, "count": int}],
            "merged": {qid: {ttype: merged_annotation}},
            "disagreements": [{"qid", "ttype", "annotations": [...]}],
            "stats": {...}
        }
    """
    # Load all sources
    sources = []
    all_annotations: List[Dict[str, Any]] = []

    for path in files:
        try:
            data = load_annotation_file(path)
            count = sum(
                1
                for types in data["annotations"].values()
                for t in types.values()
                if isinstance(t, dict) and t.get("judgment")
            )
            sources.append(
                {
                    "assignee": data["assignee"],
                    "file": path.name,
                    "count": count,
                }
            )
            all_annotations.append(data)
            logger.info(
                f"로드: {path.name} (작업자: {data['assignee']}, {count}개 판정)"
            )
        except Exception as e:
            logger.warning(f"로드 실패: {path} — {e}")

    if not all_annotations:
        logger.error("로드된 annotation 파일 없음")
        sys.exit(1)

    # Merge
    merged: Dict[str, Dict[str, Any]] = {}
    disagreements: List[Dict[str, Any]] = []

    # Collect all (qid, ttype) pairs
    all_qid_ttypes: Dict[str, Dict[str, List[Dict]]] = {}
    for data in all_annotations:
        assignee = data["assignee"]
        for qid, types in data["annotations"].items():
            if qid not in all_qid_ttypes:
                all_qid_ttypes[qid] = {}
            for ttype, tdata in types.items():
                if not isinstance(tdata, dict):
                    continue
                if ttype not in all_qid_ttypes[qid]:
                    all_qid_ttypes[qid][ttype] = []
                entry = {**tdata, "assignee": assignee}
                all_qid_ttypes[qid][ttype].append(entry)

    # Resolve each (qid, ttype)
    stats = {
        "total_items": 0,
        "single_reviewer": 0,
        "agreed": 0,
        "disagreed": 0,
        "edited": 0,
    }

    for qid in sorted(all_qid_ttypes.keys()):
        merged[qid] = {}
        for ttype in sorted(all_qid_ttypes[qid].keys()):
            entries = all_qid_ttypes[qid][ttype]
            stats["total_items"] += 1

            # Filter entries with actual judgments
            judged = [e for e in entries if e.get("judgment")]

            if not judged:
                # Only edits, no judgments — take latest edit
                edited = [e for e in entries if e.get("edited_context")]
                if edited:
                    latest = max(edited, key=lambda e: e.get("timestamp", ""))
                    merged[qid][ttype] = latest
                    stats["edited"] += 1
                continue

            if len(judged) == 1:
                # Single reviewer
                merged[qid][ttype] = judged[0]
                stats["single_reviewer"] += 1
                if judged[0].get("edited_context"):
                    stats["edited"] += 1
                continue

            # Multiple reviewers — check agreement
            judgments = set(e["judgment"] for e in judged)

            if len(judgments) == 1:
                # Agreed — take the one with most detail (note/edit)
                best = max(
                    judged,
                    key=lambda e: (
                        bool(e.get("edited_context")),
                        bool(e.get("note")),
                        e.get("timestamp", ""),
                    ),
                )
                best["agreed_by"] = [e["assignee"] for e in judged]
                merged[qid][ttype] = best
                stats["agreed"] += 1
            else:
                # Disagreement — flag for manual resolution
                disagreements.append(
                    {
                        "qid": qid,
                        "ttype": ttype,
                        "annotations": judged,
                    }
                )
                # Temporarily take the latest
                latest = max(judged, key=lambda e: e.get("timestamp", ""))
                latest["needs_resolution"] = True
                latest["all_judgments"] = [
                    {"assignee": e["assignee"], "judgment": e["judgment"]}
                    for e in judged
                ]
                merged[qid][ttype] = latest
                stats["disagreed"] += 1

            if any(e.get("edited_context") for e in judged):
                stats["edited"] += 1

    return {
        "merged_at": datetime.now().isoformat(),
        "sources": sources,
        "merged": merged,
        "disagreements": disagreements,
        "stats": stats,
    }


def print_report(result: Dict[str, Any]) -> None:
    """Print merge summary to console."""
    stats = result["stats"]
    sources = result["sources"]
    disagreements = result["disagreements"]

    logger.info(f"\n{'=' * 60}")
    logger.info("MERGE REPORT")
    logger.info(f"{'=' * 60}")

    logger.info(f"\n작업자 ({len(sources)}명):")
    for s in sources:
        logger.info(f"  {s['assignee']}: {s['count']}개 판정 ({s['file']})")

    logger.info("\n통계:")
    logger.info(f"  전체 항목:     {stats['total_items']}")
    logger.info(f"  단일 리뷰어:   {stats['single_reviewer']}")
    logger.info(f"  판정 일치:     {stats['agreed']}")
    logger.info(f"  판정 불일치:   {stats['disagreed']}")
    logger.info(f"  직접 편집:     {stats['edited']}")

    if disagreements:
        logger.info(f"\n판정 불일치 ({len(disagreements)}건):")
        for d in disagreements:
            judgments_str = ", ".join(
                f"{a['assignee']}={a['judgment']}" for a in d["annotations"]
            )
            logger.info(f"  {d['qid']}/{d['ttype']}: {judgments_str}")
            # Show notes if any
            for a in d["annotations"]:
                if a.get("note"):
                    logger.info(f"    {a['assignee']} 메모: {a['note'][:80]}")


def main():
    parser = argparse.ArgumentParser(description="Merge annotation files")
    parser.add_argument(
        "--dir",
        type=str,
        default="experiments/results/metacognitive/annotations",
        help="Directory containing annotation JSON files",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Specific files to merge (relative to --dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for merged JSON",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    ann_dir = project_root / args.dir

    if args.files:
        files = [ann_dir / f for f in args.files]
    else:
        if not ann_dir.exists():
            logger.error(f"디렉토리 없음: {ann_dir}")
            logger.info("작업자 JSON을 해당 폴더에 저장 후 다시 실행하세요.")
            sys.exit(1)
        files = sorted(ann_dir.glob("*.json"))
        # Exclude merged.json and disagreements.json
        files = [f for f in files if f.stem not in ("merged", "disagreements")]

    if not files:
        logger.error("머지할 JSON 파일 없음")
        sys.exit(1)

    logger.info(f"머지 대상: {len(files)}개 파일")
    result = merge_annotations(files)
    print_report(result)

    # Save merged
    if args.output:
        output_path = project_root / args.output
    else:
        output_path = ann_dir / "merged.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "merged_at": result["merged_at"],
                "sources": result["sources"],
                "stats": result["stats"],
                "annotations": result["merged"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"\n머지 결과 저장: {output_path}")

    # Save disagreements separately if any
    if result["disagreements"]:
        disagree_path = output_path.parent / "disagreements.json"
        with open(disagree_path, "w", encoding="utf-8") as f:
            json.dump(result["disagreements"], f, ensure_ascii=False, indent=2)
        logger.info(f"불일치 목록 저장: {disagree_path}")


if __name__ == "__main__":
    main()
