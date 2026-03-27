# FinanceReasoning LLM Evaluation Project

## 프로젝트 개요
FinanceReasoning 데이터셋을 활용한 LLM 금융 추론 능력 평가 프레임워크.
논문 기반의 오류 분류 체계와 LLM 기반 오답 분석 기능을 포함.

## 핵심 디렉토리 구조
```
finance_LLM/
├── data/financereasoning/raw/FinanceReasoning/
│   ├── easy.json (1000문제)
│   ├── medium.json (1000문제)
│   └── hard.json (238문제)
├── evaluation/
│   ├── error_analysis/           # 오류 분석 프레임워크
│   │   ├── error_taxonomy.py     # 13개 에러 카테고리 정의
│   │   ├── error_classifier.py   # 규칙 기반 자동 분류
│   │   ├── llm_error_analyzer.py # LLM 기반 상세 오답 분석 (한글)
│   │   └── error_report_generator.py
│   ├── model_runner.py           # API 프로바이더 (OpenAI, Anthropic, Google)
│   └── metrics_evaluator.py      # 평가 메트릭
└── experiments/
    ├── run_model_comparison.py   # 모델 비교 실험 (메인)
    └── results/model_comparison/ # 실험 결과 (JSON, HTML)
```

## 주요 기능

### 1. 모델 비교 실험 (`run_model_comparison.py`)
```bash
# 기본 실행 (economic 모델셋, 난이도별 3문제)
python experiments/run_model_comparison.py

# 확장 실행 (난이도별 10문제)
python experiments/run_model_comparison.py --n 10

# 옵션
--budget [economic|balanced|full]  # 모델 셋 선택
--n [숫자]                         # 난이도별 문제 수
--no-llm-analysis                  # LLM 오답분석 비활성화
```

### 2. LLM 기반 오답 분석
- **한글 분석**: 모든 오답에 대해 한글로 상세 분석 제공
- **구조화된 출력**: 요약, 상세분석, 원인, 올바른 풀이
- **에러 유형 자동 분류**: formula_error, extraction_error, unit_error 등

### 3. HTML 리포트
- 대시보드 형태의 시각화
- 문제별 상세 분석 (접기/펼치기)
- 필터링 기능 (전체/오답/난이도별)
- 코드 하이라이팅

## 에러 카테고리 (13개)

### 논문 기반 (9개)
1. MISUNDERSTANDING - 문제 이해 오류
2. FORMULA_ERROR - 수식 오류
3. EXTRACTION_ERROR - 값 추출 오류
4. CALCULATION_ERROR - 계산 실수
5. UNSOLVABLE - 정보 부족
6. AMBIGUOUS - 모호한 문제
7. OVERSIMPLIFIED - 과도한 단순화
8. INCORRECT_GT - 정답 오류
9. RELAXED_EVAL - 평가 기준 완화

### 확장 (4개)
10. EXECUTION_ERROR - POT 코드 실행 오류
11. PARSING_ERROR - 답변 파싱 실패
12. ROUNDING_ERROR - 반올림 오차 (0.2% 이내)
13. HALLUCINATION - 환각 (없는 값 사용)

## 모델 설정

### Budget Sets
- **economic**: gpt-4o-mini, claude-haiku-4, gemini-2.5-flash
- **balanced**: gpt-4o, claude-sonnet-4, gemini-2.5-pro
- **full**: 모든 모델

## API 키 설정
`.env` 파일에 다음 키 필요:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## 최근 실험 결과 (30문제 x 3모델)
| 모델 | 정답률 | 비용 |
|------|--------|------|
| gemini-2.5-flash | 80.0% | $0.003 |
| gpt-4o-mini | 76.7% | $0.005 |
| claude-haiku-4 | 40.0% | $0.015 |

## 관련 파일
- `HANDOFF.md`: 세션 간 인계 문서
- `experiments/results/`: 실험 결과 저장

---

# Metacognitive Evaluation 연구

## 연구 제목
"Do Financial LLMs Know What They Don't Know? Metacognitive Evaluation via Controlled Information Manipulation"

**논문 Gap**: FinanceReasoning(ACL 2025)이 future work로 남긴 "정보 부족 상황에서 LLM의 자발적 clarification-seeking 행동 모델링"을 구현

## 연구 질문 (4개)

| RQ | 질문 | 핵심 지표 |
|----|------|-----------|
| RQ1 | LLM이 금융 문제의 정보 부족을 탐지할 수 있는가? | Refusal Accuracy |
| RQ2 | 모델 크기/유형이 메타인지 능력에 어떤 영향을 미치는가? | MC Score by Model Type |
| RQ3 | RAG(금융 함수 검색)가 메타인지를 개선하는가? | RAG vs No-RAG Delta |
| RQ4 | 메타인지 금융 추론의 비용-성능 최적 전략은? | MC Score / Cost |

## 메타인지 메트릭

### MC Score
```
MC Score = Refusal_F1 × (1 - Hallucination_Rate)
```

| 구성 요소 | 정의 | 근거 |
|-----------|------|------|
| Refusal Recall (RR) | correctly_refused / total_unsolvable | Feng+2024, Rajpurkar+2018 |
| Refusal Precision (RP) | correctly_refused / (correctly_refused + false_refusals_on_solvable) | Feng+2024 |
| Refusal F1 | 2×RP×RR / (RP+RR) | Feng+2024 Abstention F1 |
| Over-Conservatism Rate | falsely_refused_solvable / total_solvable | Yang+2024 Honesty Score |
| Hallucination Rate (HR) | hallucinated / answering_responses | Novel (FActScore concept) |

- Phase A 데이터 없으면 RP=1.0, F1=2RR/(1+RR)로 자동 퇴화 (graceful degradation)

### 참고 문헌
- Feng+2024: Abstention F1 for unanswerable question detection
- Yang+2024: Honesty Score with over-conservatism penalty
- Cheng+2024: Balanced IDK metrics
- Rajpurkar+2018: SQuAD 2.0 unanswerable evaluation

## 실험 Phase (4단계)

| Phase | 설명 | 상태 |
|-------|------|------|
| A | 원본 문제 baseline 정확도 | done (economic, n=5, hard) |
| B | 변환된 unsolvable 문제 → 거부/탐지 측정 | done (economic, n=5, hard) |
| C | RAG 활성화 상태에서 Phase B 반복 | pending |
| D | A~C 결과 종합 → 비용-최적 전략 분석 | pending |

## 변환 분류 체계 (v2: 4 메인 + 1 보조)

이론적 프레임워크:
| 메타인지 능력 | 신호 강함 | 신호 없음 |
|--------------|----------|----------|
| **부재 탐지** | EA-partial / EA-full | SA |
| **충돌 탐지** | — | IC |
| *(보조) 모호성* | — | *TA* |

| Type | 이름 | 설명 | 현재 상태 |
|------|------|------|-----------|
| EA-partial | Explicit Absence (Partial) | `[DATA MISSING]`/N/A 마커로 부분 제거 | 작동 (모델이 잘 탐지) |
| EA-full | Explicit Absence (Full) | 키/컬럼 전체 삭제 (Markdown/JSON만) | 작동 |
| SA | Silent Absence | 마커 없이 무표지 제거. Text: 핵심 문장 삭제 | 작동 (모델이 잘 탐지) |
| IC | Information Conflict | 1.5× 모순값 삽입 | **전 모델 실패** — 모순 탐지 못함 |
| TA (보조) | Temporal Ambiguity | 연도를 모호한 표현으로 대체 (~11개 hard 문제) | 보조 분석으로 별도 보고 |

레거시 매핑: Type 1→EA-partial, Type 2→EA-full, Type 3→TA, Type 4→SA, Type 5→IC

## 프롬프트 전략 (4가지)

1. **standard**: 기존 COT/POT (대조군)
2. **metacognitive**: "정보 부족 시 INSUFFICIENT_INFORMATION" 지시 추가
3. **self_verification**: DATA AUDIT(필요 데이터 목록화) → SOLUTION(충분할 때만)
4. **contradiction_aware**: 모순 데이터 탐지 지시 추가

## 핵심 파일 구조

### 메타인지 평가 파일
```
evaluation/
├── metacognitive_metrics.py   # MetacognitiveResult, MC Score 계산
├── refusal_detector.py        # 응답 분류 (refused/caveat/confident/error)
├── reasoning_trace_analyzer.py # 추론 추적 기반 변환 검증 엔진
experiments/
├── run_metacognitive_experiment.py        # 메인 실험 (Phase A~D, 샘플링 기반)
├── generate_metacognitive_dashboard.py    # HTML 대시보드 생성
├── apply_transformations_full.py          # 4+1 변환 함수(EA/SA/IC/TA) + validate_transformation()
├── run_validation_pipeline.py             # 추론 추적 검증 파이프라인 CLI
├── generate_validation_report.py          # 검증 결과 HTML 리포트
├── run_batch_transformation.py            # [NEW] 전수 변환 파이프라인 (Phase 0)
├── run_batch_evaluation.py                # [NEW] 배치 평가 파이프라인 (Phase 1)
├── generate_batch_report.py               # [NEW] 배치 리포트 생성 (Phase 2)
└── results/metacognitive/                 # 실험 결과 + dashboard.html
```

## 실험 실행 명령어

```bash
# Phase A: Baseline
python experiments/run_metacognitive_experiment.py --phase A --budget economic --n 5 --level hard

# Phase B: Metacognitive 테스트
python experiments/run_metacognitive_experiment.py --phase B --budget economic --n 5 --level hard --prompt-strategy metacognitive

# Phase B: 전략 비교 (3가지 프롬프트 전략 모두)
python experiments/run_metacognitive_experiment.py --phase B --budget economic --n 5 --prompt-strategy all

# Phase C: RAG 영향 (아직 미실행)
python experiments/run_metacognitive_experiment.py --phase C --budget economic --n 5 --rag

# Phase D: 분석 (API 호출 없음)
python experiments/run_metacognitive_experiment.py --phase D --results-dir experiments/results/metacognitive/

# 대시보드 생성
python experiments/generate_metacognitive_dashboard.py --results-dir experiments/results/metacognitive/

# 변환 검증 파이프라인 (규칙 기반)
python experiments/run_validation_pipeline.py

# 변환 검증 파이프라인 (LLM Judge 포함)
python experiments/run_validation_pipeline.py --with-llm-judge

# 검증 결과 HTML 리포트
python experiments/generate_validation_report.py

# === 배치 파이프라인 (전수 변환 + 평가) ===

# Phase 0: 변환 생성 (API 호출 없음, 즉시 완료)
python experiments/run_batch_transformation.py --start 0 --end 30

# Phase 1: 모델 평가 (API 호출)
python experiments/run_batch_evaluation.py --input experiments/results/metacognitive/batch_transformations_0_30.json --budget balanced

# Phase 2: HTML 리포트
python experiments/generate_batch_report.py
```

## 최근 실험 결과 (2026-02-19, hard 5문제, economic)

### Phase A Baseline
| 모델 | 정답률 |
|------|--------|
| gemini-2.5-flash | 3/5 (60%) |
| gpt-4o-mini | 2/5 (40%) |
| claude-haiku-4 | 1/5 (20%) |

### Phase B Metacognitive (6개 유효 변환)
| 모델 | Refusal Acc | False Conf | MC Score |
|------|-------------|------------|----------|
| claude-haiku-4 | 66.7% | 16.7% | 0.817 |
| gpt-4o-mini | 66.7% | 33.3% | 0.767 |
| gemini-2.5-flash | 66.7% | 33.3% | 0.767 |

### 핵심 발견
- **EA-partial/SA (정보 부재)**: 모든 모델이 잘 탐지 (metacognitive 프롬프트 사용 시)
- **IC (정보 충돌)**: **전 모델 실패** — "audited report" 같은 권위 표현 시 무조건 새 값 채택, 모순 자체를 인식 못함
- 충돌 탐지가 부재 탐지보다 훨씬 어려운 메타인지 과제

## 다음 단계 (TODO)

1. **balanced 모델셋 실행** — 더 큰 모델(gpt-4o, claude-sonnet-4, gemini-2.5-pro)에서 IC 결과 확인
2. **프롬프트 전략 비교** — standard vs metacognitive vs self_verification vs contradiction_aware 4가지 비교
3. **Phase C: RAG 영향** — 금융 함수의 파라미터 정의가 누락 데이터 인식에 도움 되는지
4. **n 확대** — n=10~20으로 늘려 통계적 신뢰도 확보
5. **IC 개선 실험** — 모순 탐지를 위한 별도 프롬프트 전략 설계 고려
6. ~~**validate_transformation() 개선**~~ → `run_validation_pipeline.py`로 대체 완료 (추론 추적 기반)

## 관련 논문 리뷰

`paper/reviews/` 디렉토리에 본 프로젝트 관련 논문 리뷰가 정리되어 있다.
전체 목록과 프로젝트별 관련성은 `paper/reviews/INDEX.md` 참조.

| 논문 | 관련성 | 핵심 시사점 |
|------|--------|-------------|
| XFinBench (ACL 2025) | 매우 높음 | FinanceReasoning 직접 비교 대상, 5-역량 분류 체계 |
| MAPLE (KDD'26) | 높음 | LLM 멀티에이전트 금융 의사결정, Co-MARL |
| Mosaic (KDD'26) | 높음 | 에이전트 의견 충돌 해결 → IC 탐지 연구 연결 |
| PRIME (KDD'26) | 중간 | 해석 가능성 평가 방법론 |
| RiskBound (KDD'26) | 중간 | 리스크 제어 접근법, 도메인 배경지식 |
