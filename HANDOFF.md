# HANDOFF - 세션 인계 문서

**마지막 업데이트**: 2026-02-19
**현재 상태**: Metacognitive Evaluation 연구 - Phase A/B 완료, Phase C/D 대기

---

## 연구 개요

**제목**: "Do Financial LLMs Know What They Don't Know? Metacognitive Evaluation via Controlled Information Manipulation"

**핵심 아이디어**: FinanceReasoning(ACL 2025)의 금융 문제를 5가지 방식으로 변환하여 unsolvable하게 만든 뒤, LLM이 정보 부족을 자발적으로 인식(거부)하는지 측정

---

## 이번 세션에서 완료한 작업

### 1. 메타인지 평가 프레임워크 구축 (신규 4파일 + 기존 2파일 수정)

| 파일 | 역할 |
|------|------|
| `evaluation/metacognitive_metrics.py` | MetacognitiveResult 데이터클래스, MC Score 계산 |
| `evaluation/refusal_detector.py` | 응답 분류 (refused/caveat/confident/error), 패턴 매칭 |
| `experiments/run_metacognitive_experiment.py` | 메인 실험 (Phase A~D), 3가지 프롬프트 전략 |
| `experiments/generate_metacognitive_dashboard.py` | HTML 대시보드 + 상세 케이스 분석 |
| `experiments/apply_transformations_full.py` | Type 4/5 변환 추가, validate_transformation() |
| `evaluation/error_analysis/error_taxonomy.py` | METACOGNITIVE_REFUSAL/CAVEAT 카테고리 추가 |

### 2. Phase A/B 실험 실행 (economic, n=5, hard)

**Phase A 결과** (Baseline):
| 모델 | 정답률 |
|------|--------|
| gemini-2.5-flash | 3/5 (60%) |
| gpt-4o-mini | 2/5 (40%) |
| claude-haiku-4 | 1/5 (20%) |

**Phase B 결과** (Metacognitive, 6개 유효 변환):
| 모델 | Refusal Acc | False Conf | MC Score |
|------|-------------|------------|----------|
| claude-haiku-4 | 66.7% | 16.7% | 0.817 |
| gpt-4o-mini | 66.7% | 33.3% | 0.767 |
| gemini-2.5-flash | 66.7% | 33.3% | 0.767 |

### 3. Type 5 변환 수정 (핵심 수정)

**문제**: 기존 Type 5는 context 끝에 `[Note: ...]` 추가 → 모델이 무시
**수정**: context 본문 내부에 모순 문장 삽입 (예: "However, according to the audited financial statements, the revenue was $37,500, not $25,000.")
**결과**: 수정 후에도 **전 모델 실패** → 모순 탐지가 누락 탐지보다 훨씬 어려운 과제임을 확인

### 4. 대시보드 상세 케이스 뷰 추가

- 원본 vs 변환된 context 나란히 비교
- 모델별 응답 + 색상 코딩 (거부=녹색, 확신=빨간색, 경고=노란색)
- 변환 설명, 정답, hallucinated values 표시

---

## 핵심 발견

1. **Type 1/4 (정보 제거)**: metacognitive 프롬프트 사용 시 모든 모델이 잘 탐지
2. **Type 5 (모순 정보)**: **전 모델 실패** — "audited financial statements"라는 권위 표현을 보면 새 값을 무조건 채택, 모순 자체를 인식 못함
3. 모순 탐지 >> 누락 탐지 난이도 (메타인지 측면)

---

## 결과 파일 위치

```
experiments/results/metacognitive/
├── phase_A_20260219_172158.json          # Baseline 결과
├── phase_B_metacognitive_20260219_172514.json  # MC 테스트 결과
└── dashboard.html                        # 시각화 대시보드
```

---

## 다음 작업 (우선순위 순)

### 즉시 실행 가능
1. **balanced 모델셋 실행** — gpt-4o, claude-sonnet-4, gemini-2.5-pro에서 Type 5 결과 확인
   ```bash
   python experiments/run_metacognitive_experiment.py --phase A --budget balanced --n 5 --level hard
   python experiments/run_metacognitive_experiment.py --phase B --budget balanced --n 5 --prompt-strategy metacognitive
   ```

2. **프롬프트 전략 비교** — standard vs metacognitive vs self_verification
   ```bash
   python experiments/run_metacognitive_experiment.py --phase B --budget economic --n 5 --prompt-strategy all
   ```

3. **n 확대** — n=10~20으로 통계적 신뢰도 확보

### 추후 작업
4. **Phase C: RAG 영향** — 금융 함수 파라미터 정의가 누락 데이터 인식에 도움 되는지
5. **Type 5 개선 실험** — 모순 탐지를 위한 별도 프롬프트 전략 설계
6. **validate_transformation() 개선** — python_solution이 context와 독립적으로 하드코딩된 값 사용하는 문제

---

## 알려진 이슈

1. **Claude Haiku 529 overloaded**: 간헐적 발생, 재시도하면 복구
2. **validate_transformation() 한계**: test-2000 등 python_solution이 하드코딩된 값 사용 → 모든 변환이 still_solvable로 판정
3. **Type 5 text 변환**: 모순 문장이 context 내부에 삽입되지만 모든 모델이 탐지 실패

---

## 빠른 시작 (새 세션용)

```bash
# 1. 환경 확인
cd C:\Users\fanding\PycharmProjects\finance_LLM

# 2. 기존 결과 확인
# experiments/results/metacognitive/dashboard.html 열기

# 3. 이어서 실험 (예: balanced 모델셋)
python experiments/run_metacognitive_experiment.py --phase A --budget balanced --n 5 --level hard
python experiments/run_metacognitive_experiment.py --phase B --budget balanced --n 5 --prompt-strategy metacognitive

# 4. 대시보드 재생성
python experiments/generate_metacognitive_dashboard.py --results-dir experiments/results/metacognitive/
```

---

## 연구 질문 (참고)

| RQ | 질문 | 핵심 지표 |
|----|------|-----------|
| RQ1 | LLM이 금융 문제의 정보 부족을 탐지할 수 있는가? | Refusal Accuracy |
| RQ2 | 모델 크기/유형이 메타인지 능력에 어떤 영향을 미치는가? | MC Score by Model Type |
| RQ3 | RAG가 메타인지를 개선하는가? | RAG vs No-RAG Delta |
| RQ4 | 비용-성능 최적 전략은? | MC Score / Cost |

---

## 메트릭 공식

- **Refusal Accuracy** = 정확히 거부한 unsolvable 문제 / 전체 unsolvable
- **False Confidence Rate** = 확신있게 답한 unsolvable 문제 / 전체 unsolvable
- **MC Score** = 0.4 x RefusalAcc + 0.3 x (1-FalseConf) + 0.3 x (1-HalluRate)
