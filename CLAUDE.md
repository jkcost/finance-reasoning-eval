# FinanceReasoning LLM Evaluation Project

## 프로젝트 개요
FinanceReasoning 데이터셋을 활용한 LLM 금융 추론 능력 평가 프레임워크.
논문 기반의 오류 분류 체계와 LLM 기반 오답 분석 기능을 포함.

## 코드 관리 원칙 (CRITICAL)
- **코드 수정/실행 후 불필요한 파일 정리**: 로직이 바뀌어 더 이상 사용하지 않는 코드, 이전 실험 결과 파일은 즉시 삭제하거나 `_archive/`로 이동
- **폴더를 항상 깨끗하게 유지**: 팀원이 clone했을 때 현재 활성 파일만 보여야 함
- **작업 이력은 MEMORY에 기록**: 삭제/변경된 파일의 목적과 변경 이유를 memory에 저장하여 에이전트가 맥락을 유지
- **`_archive/`는 .gitignore**: 로컬에만 보관, Git에는 올리지 않음
- **`experiments/results/`는 .gitignore**: 실험 결과는 Git에 올리지 않음 (annotations/ 제외)

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

## 연구 질문 (5개)

| RQ | 질문 | 핵심 지표 |
|----|------|-----------|
| RQ1 | LLM은 정보 부재와 정보 충돌에서 비대칭적 메타인지 능력을 보이는가? | Refusal Rate: absence vs IC |
| RQ2 | 모델 크기/유형이 충돌 탐지 능력에 어떤 영향을 미치는가? | IC detection rate by tier |
| RQ3 | 프롬프트 전략이 충돌 탐지를 개선할 수 있는가? | IC detection by strategy |
| RQ4 | IC 난이도 레벨(L1-L4)이 탐지 실패를 예측하는가? | L1-L4 gradient |
| RQ5 | 인간 금융 전문가 대비 LLM의 충돌 탐지 능력은? | Human vs LLM |

Note: 기존 RQ3 (RAG)는 본 논문 scope에서 제외. 기존 RQ4 (비용분석)는 Phase D로 이동.

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
| 0 | 238문제 전수 LLM 변환 (8타입) | done (gemini-2.5-flash, 98.9% 성공) |
| Human Review | 변환 품질 검증 (4명 분배) | 진행 중 |
| A | 원본 문제 baseline 정확도 | done (economic, pilot) |
| B | 변환된 문제 → 거부/탐지 측정 | done (economic pilot), balanced 대기 |
| D | 종합 분석 | balanced 실험 후 |

## 변환 분류 체계 (v2: 4 메인 + 1 보조)

이론적 프레임워크:
| 메타인지 능력 | 신호 강함 | 신호 없음 |
|--------------|----------|----------|
| **부재 탐지** | EA-partial / EA-full | SA |
| **충돌 탐지** | — | IC |
| *(보조) 모호성* | — | *TA* |

Absence Detection (정보 부재):
| Type | 이름 | 설명 |
|------|------|------|
| EA-partial | Explicit Absence (Partial) | 값을 N/A 마커로 교체 |
| EA-full | Explicit Absence (Full) | 키/컬럼 전체 삭제 |
| SA | Silent Absence | 마커 없이 무표지 제거 |

Conflict Detection (정보 충돌) — IC Difficulty Ladder:
| Level | 이름 | 설명 |
|-------|------|------|
| IC-L1 | 10x 타이포 | 숫자를 10배 틀리게 교체 |
| IC-L2 | 단위 불일치 | 같은 값을 다른 단위로 삽입 + 1.5x 오류 |
| IC-L3 | 권위 충돌 | 권위 있는 출처가 1.5x 모순값 제시 |
| IC-L4 | 기간 합산 불일치 | 분기합 ≠ 연간총계 |

보조:
| TA | Temporal Ambiguity | 연도를 모호한 표현으로 대체 (보조, TA는 LLM 파이프라인에서 skip) |

레거시 매핑: Type 1→EA-partial, Type 2→EA-full, Type 3→TA, Type 4→SA, Type 5→IC

## 프롬프트 전략 (6가지)

1. standard: 기존 COT/POT
2. metacognitive: "정보 부족 시 INSUFFICIENT_INFORMATION" 지시
3. self_verification: DATA AUDIT → SOLUTION 2단계
4. contradiction_aware: 모순 데이터 탐지 3단계
5. ic_fewshot: IC 탐지용 few-shot 예시 제공 (mitigation)
6. ic_crosscheck: 모든 수치 교차 검증 의무화 (mitigation)

## 핵심 파일 구조

### 핵심 파일
```
evaluation/
├── metacognitive_metrics.py        # MC Score 계산
├── refusal_detector.py             # 응답 분류 (refused/caveat/confident/error)
├── reverse_calc_detector.py        # EA-full 역산 가능성 자동 탐지
├── reasoning_trace_analyzer.py     # 추론 추적 분석
experiments/
├── llm_transform.py                # LLM 기반 변환 엔진 (핵심)
├── ic_difficulty_ladder.py         # IC L1-L4 규칙 기반 변환 (보조)
├── conflict_salience_scorer.py     # IC salience 회귀분석 도구
├── run_batch_transformation.py     # 전수 LLM 변환 파이프라인
├── run_batch_evaluation.py         # 배치 평가 (checkpoint/resume 지원)
├── run_metacognitive_experiment.py # 메인 실험 (Phase A~D)
├── generate_human_review.py        # Human Review HTML 생성 (8타입 + 모델 응답)
├── generate_review_summary.py      # 팀 토론용 요약 HTML
├── human_baseline_study.py         # 인간 비교 설문 생성
├── apply_transformations_full.py   # 변환 로직 (레거시 규칙 기반)
├── merge_annotations.py            # Annotation 머지
└── results/metacognitive/          # 실험 결과 (.gitignore)
    └── annotations/                # 리뷰어 JSON
tests/
├── test_ic_difficulty_ladder.py    # IC L1-L4 테스트 (17개)
└── test_conflict_salience_scorer.py # Salience 테스트 (8개)
paper/
├── paper_outline.md                # 논문 구조 + 실험 목록
└── reviews/                        # 관련 논문 리뷰 (7편)
```

## 실험 실행 명령어

```bash
# === 현재 워크플로우 ===

# 1. LLM 변환 (238문제 전체, API 필요)
python experiments/run_batch_transformation.py --start 0 --end 238

# 2. Human Review HTML 생성 (모델 응답 포함)
python experiments/generate_human_review.py \
    --input batch_transformations_0_238.json \
    --eval batch_evaluation_0_238.json

# 3. 모델 평가 (checkpoint/resume 지원, API 필요)
python experiments/run_batch_evaluation.py \
    --input experiments/results/metacognitive/batch_transformations_0_238.json \
    --budget balanced --prompt-strategy metacognitive

# 4. 팀 토론 요약 (리뷰 완료 후)
python experiments/generate_review_summary.py

# 5. 테스트
python -m pytest tests/ -v
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
- 정보 부재 탐지 (EA/SA): 모델이 비교적 잘 인식 (~67% 거부율)
- 정보 충돌 탐지 (IC): 모델/전략/난이도에 따라 극적 차이 (0%~80%)
- IC 탐지는 "전 모델 실패"가 아니라 "조건부 실패" — 어떤 조건에서 성공/실패하는가가 핵심 연구 질문

## 변환 구현 원칙 (CRITICAL)

**변환은 반드시 LLM 기반으로 수행한다.**
- 규칙 기반(regex, 키워드 매칭)은 금융 맥락을 이해하지 못해 부적절한 변환을 생성함
- LLM이 문제를 분석하고, 풀이에 필수적인 데이터를 식별한 뒤, 변환 기준에 따라 변환을 생성해야 함
- `apply_transformations_full.py`의 기존 규칙 기반 함수는 v1 레거시이며, LLM 기반 파이프라인으로 교체 예정
- 변환 생성 시 LLM이 판단해야 할 것:
  1. "이 문제를 풀려면 어떤 데이터가 필수적인가?"
  2. "이 데이터를 제거하면 다른 방법으로 도출 가능한가?" (역산 가능성 체크)
  3. "자연스럽게 제거/변형하려면 어떻게 바꿔야 하는가?"

## Human Review 협업 도구

- `experiments/generate_human_review.py`: 변환 결과 리뷰 HTML 생성 (v2)
- `experiments/merge_annotations.py`: 작업자별 annotation 머지 + 불일치 감지
- `docs/REVIEW_GUIDE.md`: 협업 가이드
- 작업자 분배: URL 파라미터 `?assignee=이름&start=N&end=M`
- annotation 저장: `experiments/results/metacognitive/annotations/`

## 제출 전략 (2026-04-15 결정)

**Primary target: CIKM 2026 Short (4p, 마감 2026-06-06, AoE, Rome 11/7~11/11)**
**Follow-up target: AAAI 2027 (7p, 마감 2026-08-01 / abstract 7/25, Montréal 2027-02)**

### CIKM Short RQ 3축 (확장)

| RQ | 내용 | 산출물 |
|----|------|--------|
| **RQ-A (핵심 finding)** | FinanceReasoning에서 LLM의 absence vs conflict 비대칭성 및 IC L1-L4 gradient | Phase A/B balanced 결과 + MC Score 표 + L1-L4 gradient figure |
| **RQ-B (POT Faithfulness 정량)** | POT 응답의 숫자가 변환된 context에서 왔는가, 원본 회상인가? Value Provenance Classifier + Memorization Score 도입 | `value_provenance_classifier.py` + 모델별 Memorization Score 표 |
| **RQ-C (Conflict Salience Regression)** | IC 탐지 성공/실패를 예측하는 feature(token_distance, authority_marker, magnitude_ratio, same_paragraph)는 무엇인가? | `conflict_salience_scorer.py` 확장 + logistic regression AUC + feature coefficient 표 |

### AAAI 2027 확장 축 (CIKM 이후 8주)

- **RQ-D (Cross-dataset 일반화)**: FinQA(또는 TAT-QA)에 taxonomy 이식 → "단일 도메인" 리뷰어 반박
- **RQ-E (Human Baseline)**: 금융 전문가 5~10명 IC-L3 detection 측정 → "사람은 할 수 있다" 증명
- **Error Pattern Taxonomy**: 실패 mode 세분화 (무시/암묵채택/인지후진행)

## 남은 52일 주차별 계획 (CIKM Short)

| 주차 | 기간 | 핵심 작업 |
|------|------|----------|
| W1 | 4/15~4/21 | Human Review 마무리(4명) + balanced 실험 kickoff + Value Provenance Classifier 설계 |
| W2 | 4/22~4/28 | Balanced 3모델 × 6 전략 실험 실행 (GPT-4o, Claude Sonnet, Gemini Pro) |
| W3 | 4/29~5/5 | POT faithfulness analysis + Memorization Score 계산 (RQ-B) |
| W4 | 5/6~5/12 | Conflict Salience Regression 구현 + fitting (RQ-C) + draft 시작 |
| W5 | 5/13~5/19 | Results 섹션 + figures |
| W6 | 5/20~5/26 | Related Work + Discussion + Methodology 마무리 |
| W7 | 5/27~6/2 | Co-author review + 교정 |
| W8 | 6/3~6/6 | 최종 submit |

## 논문 framing (Short 4p)

**제목 후보**: *"LLMs Detect Missing Data But Not Contradictions: A Metacognitive Evaluation in Financial Reasoning"*

**구조**:
- Introduction (0.5p) — EA vs IC 비대칭 선언
- Methodology (1p) — Transformation taxonomy + MC Score + Value Provenance Classifier + Salience features
- Experiments (1p) — Phase A/B 결과표 + IC L1-L4 gradient
- Analysis (1p) — POT faithfulness (RQ-B) + Salience regression (RQ-C) + Mitigation
- Conclusion/Limitation (0.5p)

## 정정된 레퍼런스 (Paper Verification 2026-04-15)

할루시네이션 0건 확인. 인용 시 다음 정정 적용:
- Yang+2024 "Honesty Score" → 정확 제목 *"Alignment for Honesty"*, arXiv:2312.07000
- Cheng+2024 "Balanced IDK" → 원제 *"Can AI Assistants Know What They Don't Know?"*, ICML 2024, arXiv:2401.13275
- CoT Faithfulness via Unlearning (arXiv:2502.14829) → **EMNLP 2025** Outstanding (ACL 아님)
- Memorization in APR 수치(81.8%/88.2%) → **FSE 2025 자매논문** *"Demystifying Memorization..."*에서 나옴
- ReliabilityBench (arXiv:2601.06112) → **금융 벤치마크 아님** (일반 agent reliability), 인용 제외
- CNFinBench (arXiv:2512.09506) → 부제. 본제 *"Beyond Knowledge to Agency..."*
- Lopez-Lira "S&P 500 <1% 회상" 주장 → 구체 출처 미확정, 인용 전 재확인

핵심 경쟁 논문 (정면 positioning 필요):
- **AbstentionBench** (Kirichenko et al., arXiv:2506.09038, 2025) — Meta FAIR, abstention 20 dataset. 일반 도메인/conflict 미커버
- **ConflictBank** (Su et al., NeurIPS 2024, arXiv:2408.12076) — RAG 검색 충돌 중심, intra-context 수치 충돌 약함
- **MAGIC** (Findings EMNLP 2025, aclanthology.org/2025.findings-emnlp.466) — multi-hop inter-context conflict gradient (수치 ladder 없음)

## 후속 연구 (TODOS.md 참조)
- Cross-Domain Validation (법률/의료)
- Attention Pattern Analysis (GPU 확보 시)

## 프로젝트 Memory (Git 동기화)

프로젝트 맥락·의사결정·과거 작업 이력은 `docs/memory/` 에 git-tracked 상태로 보관한다.
기기 간 동기화는 `git pull` 만으로 완료.

- `docs/memory/MEMORY.md` — 인덱스 (항상 먼저 확인)
- `docs/memory/project_submission_strategy.md` — CIKM 2026 Short → AAAI 2027 제출 전략
- `docs/memory/project_pot_faithfulness_rqb.md` — RQ-B (POT Faithfulness 정량 도구)
- `docs/memory/project_conflict_salience_rqc.md` — RQ-C (Conflict Salience Regression)
- `docs/memory/feedback_llm_based_transformation.md` — 변환은 반드시 LLM 기반
- `docs/memory/project_rule_based_v1_archive.md` — 규칙 기반 v1 아카이브

**Auto-memory 동기화**: Claude Code는 `~/.claude/projects/.../memory/`에 auto-save 하므로 세션 종료 시 `cp ~/.claude/projects/C--Users-fanding-PycharmProjects-finance-LLM/memory/*.md docs/memory/` 로 동기화 후 커밋. Mac 경로는 `/Users/{name}/.claude/projects/-Users-{name}-Projects-finance_LLM/memory/` 와 유사.

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

## gstack + Superpowers 병용 가이드

### 역할 분담
- **Superpowers**: 프로세스/방법론 (설계, 계획, TDD, 서브에이전트, Git worktree)
- **gstack**: 도구/실행 (코드 리뷰, 보안 감사, 배포, 안전 가드레일, 회고)

### 본 프로젝트 핵심 스킬
| 작업 | 사용할 스킬 | 이유 |
|------|------------|------|
| 실험 설계 | Superpowers `brainstorming` | 소크라틱 대화로 연구 방향 정제 |
| 구현 계획 | Superpowers `writing-plans` | 2~5분 단위 태스크 분해 |
| 구현 | Superpowers `subagent-driven-development` | 서브에이전트 + 2단계 리뷰 |
| TDD | Superpowers `test-driven-development` | RED-GREEN-REFACTOR |
| 코드 리뷰 | gstack `/review` + `/cso` | 보안 감사 포함 |
| 디버깅 | Superpowers `systematic-debugging` | 4단계 근본 원인 분석 |
| 배포 | gstack `/ship` | main 동기화 → 테스트 → PR |
| 안전 | gstack `/careful`, `/freeze` | 실험 데이터 보호 |
| 회고 | gstack `/retro` | 주간 작업 분석 |
| Git 관리 | Superpowers `using-git-worktrees` | 격리 개발 |

### Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.

Key routing rules:
- 연구 방향, 아이디어, brainstorming → Superpowers `brainstorming`
- 구현 계획 수립 → Superpowers `writing-plans`
- 테스트 작성 → Superpowers `test-driven-development`
- Bugs, errors, "why is this broken" → Superpowers `systematic-debugging`
- 코드 리뷰, check my diff → gstack `/review`
- 보안 감사 → gstack `/cso`
- Ship, deploy, push, create PR → gstack `/ship`
- Weekly retro → gstack `/retro`
- Architecture review → gstack `/plan-eng-review`
