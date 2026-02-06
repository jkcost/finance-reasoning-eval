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
