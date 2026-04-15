---
name: 규칙 기반 변환 v1 작업 이력 (아카이브)
description: 2026-03-27까지 진행된 규칙 기반 변환 파이프라인의 작업 이력. LLM 기반으로 교체 예정.
type: project
---

규칙 기반 변환 파이프라인 (v1) — LLM 기반으로 교체 예정

**Why:** 규칙 기반(regex, 키워드 매칭)은 금융 맥락을 이해하지 못해 부적절한 변환을 생성. 사용자가 LLM 기반을 기대했으나 규칙 기반으로 구현되어 있었음.

**작업 내역 (2026-03-27~30):**
- `apply_transformations_full.py`: 5가지 변환 유형 규칙 기반 구현 (EA-partial, EA-full, SA, IC, TA)
  - 마크다운 테이블 컬럼 정렬 버그 수정 (`_parse_markdown_table()` 추가)
  - critical value 기반 타겟 선택 (`_extract_critical_solution_values` BFS 추적)
  - Question-only 변환 함수 추가 (context 없는 19문제 대응)
- `run_batch_transformation.py`: 배치 변환 파이프라인 (0~120 문제)
- `generate_human_review.py`: 인터랙티브 HTML 리뷰 도구 (v2)
  - 변환 직접 편집, 작업자 분배, JSON 내보내기/가져오기, 키보드 단축키
  - 변환 유형 가이드 내장, Q변환 배지 구분
- `merge_annotations.py`: 작업자별 annotation 머지 + 불일치 감지
- 프로젝트 정리: legacy 26파일 → `_archive/`, temp 11파일 삭제
- `README.md`, `docs/REVIEW_GUIDE.md` 작성

**How to apply:** 이 파일들의 HTML 리뷰 도구와 merge 인프라는 LLM 기반 변환에서도 그대로 재사용. 변환 로직만 교체.
