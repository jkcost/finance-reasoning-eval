# FinanceReasoning Metacognitive Evaluation

> **"Do Financial LLMs Know What They Don't Know?"**
> LLM의 금융 추론에서 메타인지 능력(정보 부족/충돌 인식)을 평가하는 연구 프레임워크

## Quick Start

```bash
# 1. 클론
git clone https://github.com/jkcost/finance-reasoning-eval.git
cd finance-reasoning-eval
git checkout feat/metacognitive-evaluation-framework

# 2. 환경 설정
conda activate <your-env>
pip install -r evaluation/requirements.txt

# 3. API 키 설정 (.env 파일)
cp .env.example .env  # 키 입력

# 4. Human Review 시작 (아래 "현재 작업" 참조)
```

## 프로젝트 구조

```
├── data/financereasoning/raw/    # FinanceReasoning 데이터셋 (easy/medium/hard.json)
├── evaluation/                   # 핵심 평가 모듈
│   ├── model_runner.py           #   API 프로바이더 (OpenAI, Anthropic, Google)
│   ├── metacognitive_metrics.py  #   MC Score 계산
│   ├── refusal_detector.py       #   응답 분류 (refused/caveat/confident)
│   ├── reasoning_trace_analyzer.py # 추론 추적 분석
│   └── ...
├── experiments/                  # 실험 파이프라인
│   ├── apply_transformations_full.py  # 변환 로직 (EA/SA/IC/TA)
│   ├── run_batch_transformation.py    # Phase 0: 전수 변환
│   ├── run_batch_evaluation.py        # Phase 1: 모델 평가
│   ├── generate_human_review.py       # Human Review HTML 생성
│   ├── merge_annotations.py           # Annotation 머지
│   └── ...
├── docs/                         # 가이드 문서
│   └── REVIEW_GUIDE.md           # Human Review 협업 가이드
└── paper/                        # 논문 관련
    ├── research_framework.md
    └── reviews/                  # 관련 논문 리뷰
```

## 현재 작업: Human Review (변환 품질 검증)

자동 변환된 결과를 팀원들이 검토하여 품질을 확보하는 단계입니다.

**상세 가이드**: [docs/REVIEW_GUIDE.md](docs/REVIEW_GUIDE.md)

### 빠른 시작

```bash
# 1. 변환 생성 (API 호출 없음, 즉시 완료)
python experiments/run_batch_transformation.py --start 0 --end 30

# 2. 리뷰 HTML 생성
python experiments/generate_human_review.py

# 3. 브라우저에서 열기 (본인 이름과 담당 범위 지정)
# experiments/results/metacognitive/human_review_0_30.html?assignee=이름&start=0&end=10
```

## 연구 개요

### 변환 유형 (5가지)

| 유형 | 방법 | 목적 |
|------|------|------|
| **EA-partial** | 특정 값을 N/A로 교체 | 명시적 마커 탐지 (기본선) |
| **EA-full** | 컬럼/키 전체 삭제 | 구조적 부재 인식 |
| **SA** | 마커 없이 조용히 삭제 | 무표지 부재 탐지 (가장 어려움) |
| **IC** | 1.5x 모순값 삽입 | 수치 모순 탐지 (전 모델 실패) |
| **TA** | 연도 → 모호한 표현 | 시간적 모호성 (보조) |

### MC Score

```
MC Score = Refusal_F1 × (1 - Hallucination_Rate)
```

### 실험 Phase

| Phase | 설명 | 상태 |
|-------|------|------|
| A | 원본 문제 baseline | 완료 |
| B | 변환된 문제 → 거부 측정 | 완료 |
| C | RAG 활성화 상태 | 완료 |
| D | 종합 분석 | 완료 |
| **Human Review** | **변환 품질 검증** | **진행 중** |
