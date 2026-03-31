# Do Financial LLMs Know What They Don't Know?

> Metacognitive Evaluation via Controlled Information Manipulation

LLM이 금융 문제에서 **정보 부족(Absence)**과 **정보 충돌(Conflict)**을 인식하는 능력을 평가하는 연구 프레임워크입니다.

## 핵심 발견

LLM의 메타인지 능력에는 **비대칭**이 존재합니다:

- **정보 부재 탐지 (Absence Detection)**: 모델이 비교적 잘 인식 (~67% 거부율)
- **정보 충돌 탐지 (Conflict Detection)**: **모델/전략/난이도에 따라 극적 차이** (0%~80%)

이 비대칭을 체계적으로 분석하기 위해 **IC Conflict Difficulty Ladder (L1-L4)**를 도입하여, 어떤 종류의 충돌을 탐지하고 어떤 종류에서 실패하는지 연구합니다.

## 변환 체계 (Transformation Taxonomy)

원본 금융 문제에서 풀이에 필요한 데이터를 변형하여 "unsolvable" 문제를 생성합니다.

### Absence Detection (대조군)

| 유형 | 방법 | 탐지 난이도 |
|------|------|-------------|
| **EA-partial** | 특정 값을 `N/A`로 교체 | LOW — 마커가 보임 |
| **EA-full** | 컬럼/키 전체 삭제 | MODERATE — 구조 변화 |
| **SA** | 마커 없이 조용히 삭제 | HIGH — 단서 없음 |

### Conflict Detection (실험군) — IC Difficulty Ladder

| Level | 유형 | 방법 | 탐지 난이도 |
|-------|------|------|-------------|
| **IC-L1** | 10x 타이포 | 숫자를 10배 틀리게 교체 | LOW |
| **IC-L2** | 단위 불일치 | 같은 값을 다른 단위(million/billion)로 삽입 + 1.5x 오류 | MODERATE |
| **IC-L3** | 권위 충돌 | 권위 있는 출처("감사보고서")가 1.5x 모순값 제시 | HIGH |
| **IC-L4** | 기간 합산 불일치 | 분기별 값의 합계 ≠ 연간 총계 | HIGH |

### 보조

| 유형 | 방법 | 비고 |
|------|------|------|
| **TA** | 연도 → "해당 기간" | ~15문제만 적용 가능, 별도 보고 |

## MC Score (메타인지 점수)

```
MC Score = Refusal_F1 × (1 − Hallucination_Rate)
```

- Refusal Recall = 올바른 거부 / 전체 unsolvable 문제
- Refusal Precision = 올바른 거부 / 전체 거부 응답
- Refusal F1 = 2 × RP × RR / (RP + RR)

## 프로젝트 구조

```
finance_LLM/
├── data/financereasoning/raw/        # FinanceReasoning 데이터셋
│   └── FinanceReasoning/
│       ├── easy.json (1000문제)
│       ├── medium.json (1000문제)
│       └── hard.json (238문제)       # ← 본 연구 대상
│
├── evaluation/                       # 평가 모듈
│   ├── model_runner.py               #   API 호출 (OpenAI, Anthropic, Google)
│   ├── metacognitive_metrics.py      #   MC Score 계산
│   ├── refusal_detector.py           #   응답 분류 (refused/caveat/confident)
│   ├── reverse_calc_detector.py      #   EA-full 역산 가능성 자동 탐지
│   └── reasoning_trace_analyzer.py   #   추론 추적 분석
│
├── experiments/                      # 실험 파이프라인
│   ├── apply_transformations_full.py #   변환 로직 (EA/SA 기존 5타입)
│   ├── ic_difficulty_ladder.py       #   IC L1-L4 난이도 변환 모듈
│   ├── conflict_salience_scorer.py   #   IC salience 회귀분석 도구
│   ├── generate_batch_transformations.py  # 변환 생성 (규칙 기반, API 불필요)
│   ├── run_batch_evaluation.py       #   모델 평가 (checkpoint/resume 지원)
│   ├── run_metacognitive_experiment.py #  메인 실험 (Phase A~D)
│   ├── generate_human_review.py      #   Human Review HTML 생성
│   ├── generate_review_summary.py    #   팀 토론용 요약 HTML 생성
│   ├── human_baseline_study.py       #   인간 비교 설문 생성
│   └── results/metacognitive/        #   실험 결과 (.gitignore)
│       └── annotations/              #   리뷰어 annotation JSON
│
├── tests/                            # 테스트 (25개)
│   ├── test_ic_difficulty_ladder.py
│   └── test_conflict_salience_scorer.py
│
├── paper/                            # 논문 관련
│   ├── paper_outline.md              #   논문 구조 + 실험 목록
│   └── reviews/                      #   관련 논문 리뷰 (7편)
│
├── docs/
│   └── REVIEW_GUIDE.md               #   Human Review 가이드
├── CLAUDE.md                         #   프로젝트 컨텍스트
└── TODOS.md                          #   후속 연구 목록
```

## 현재 작업: Human Review

자동 변환된 결과를 팀원들이 검토하여 품질을 확보하는 단계입니다.

### 1. 환경 설정

```bash
git clone https://github.com/jkcost/finance-reasoning-eval.git
cd finance-reasoning-eval
git checkout feat/metacognitive-evaluation-framework

conda activate <your-env>
pip install -r evaluation/requirements.txt
```

### 2. 변환 생성 + 리뷰 HTML 생성

```bash
# 변환 생성 (API 호출 없음, 즉시 완료)
python experiments/generate_batch_transformations.py --start 0 --end 120

# 리뷰 HTML 생성
python experiments/generate_human_review.py --input batch_transformations_0_120.json
```

### 3. 리뷰 진행

브라우저에서 HTML을 열고, URL 파라미터로 본인의 담당 범위를 지정합니다:

| 작업자 | URL |
|--------|-----|
| 작업자A | `human_review_0_120.html?assignee=이름A&start=0&end=30` |
| 작업자B | `human_review_0_120.html?assignee=이름B&start=30&end=60` |
| 작업자C | `human_review_0_120.html?assignee=이름C&start=60&end=90` |
| 작업자D | `human_review_0_120.html?assignee=이름D&start=90&end=120` |

각 문제에서 **8개 탭** (EA-partial, EA-full, SA, IC-L1~L4, TA)을 확인하고:

- **승인**: 변환이 올바르고, 남은 정보로는 문제를 풀 수 없음
- **수정 필요**: 방향은 맞지만 개선이 필요
- **부적절**: 변환 후에도 문제를 풀 수 있거나 변환이 의미 없음

> HTML 상단의 **"변환 유형 가이드 & 리뷰 기준"**을 펼치면 각 변환 타입의 상세 설명과 리뷰 체크포인트를 확인할 수 있습니다.

#### EA-full 리뷰 팁

Python Solution의 **초록색(●) 변수**에 해당하는 칼럼이 context에서 **정말 삭제됐는지** 확인하세요. 초록 변수와 일치하는 칼럼이 아직 남아있으면 모델이 값을 읽을 수 있으므로 변환이 불충분합니다.

> ⚠️ "역산 가능 경고"가 표시된 경우: 삭제된 칼럼 값이 남은 칼럼들로 계산될 수 있다는 자동 탐지 결과입니다. 경고가 있으면 더 신중히 확인해주세요.

### 4. 리뷰 결과 제출

리뷰 완료 후 **"내보내기"** 버튼 → JSON 파일 다운로드 → 관리자에게 전달

### 5. 팀 토론

```bash
# 각 작업자의 JSON을 annotations/ 폴더에 모음
cp review_*.json experiments/results/metacognitive/annotations/

# 요약 대시보드 생성
python experiments/generate_review_summary.py

# review_summary.html을 팀 전체와 공유 → 함께 토론
```

`review_summary.html`에서 확인할 수 있는 것:
- 변환 타입별 승인율
- 리뷰어별 통계
- **"토론 필요"** 섹션: 수정필요/부적절로 판정된 건만 필터링
- 전체 120문제 × 8타입 매트릭스

### 6. 승인된 변환으로 모델 실험

```bash
# balanced 모델 (GPT-4o, Claude Sonnet, Gemini Pro) 평가
python experiments/run_batch_evaluation.py \
    --input experiments/results/metacognitive/batch_transformations_0_120.json \
    --budget balanced \
    --prompt-strategy metacognitive
```

## 프롬프트 전략 (6가지)

| 전략 | 설명 |
|------|------|
| `standard` | 기본 COT/POT (대조군) |
| `metacognitive` | "정보 부족 시 INSUFFICIENT_INFORMATION" 지시 |
| `self_verification` | DATA AUDIT → SOLUTION 2단계 |
| `contradiction_aware` | 모순 데이터 탐지 지시 (3단계) |
| `ic_fewshot` | **IC 탐지용**: 모순 예시를 few-shot으로 제공 |
| `ic_crosscheck` | **IC 탐지용**: 모든 수치 교차 검증 의무화 |

## 모델 셋

| Budget | 모델 |
|--------|------|
| economic (파일럿) | gpt-4o-mini, claude-haiku-4, gemini-2.5-flash |
| balanced (본 실험) | gpt-4o, claude-sonnet-4, gemini-2.5-pro |

## 연구 질문

| RQ | 질문 |
|----|------|
| RQ1 | LLM은 정보 부재와 정보 충돌에서 비대칭적 메타인지 능력을 보이는가? |
| RQ2 | 모델 크기/유형이 충돌 탐지 능력에 어떤 영향을 미치는가? |
| RQ3 | 프롬프트 전략이 충돌 탐지를 개선할 수 있는가? |
| RQ4 | IC 난이도 레벨(L1-L4)이 탐지 실패를 예측하는가? |
| RQ5 | 인간 금융 전문가 대비 LLM의 충돌 탐지 능력은? |

## 테스트

```bash
python -m pytest tests/ -v
```

## 관련 논문

본 연구와 관련된 논문 리뷰는 `paper/reviews/INDEX.md`를 참조하세요.

주요 참고:
- FinanceReasoning (ACL 2025) — 본 데이터셋의 원논문
- AbstentionBench (Meta, 2025) — LLM abstention 벤치마크
- "Know Your Limits" (TACL 2025) — LLM abstention 서베이
- XFinBench (ACL 2025) — 금융 추론 벤치마크

## API 키 설정

`.env` 파일에 다음 키가 필요합니다:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## 라이선스

연구 목적으로만 사용. FinanceReasoning 데이터셋의 라이선스를 따릅니다.
