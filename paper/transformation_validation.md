# Transformation Validation: Design Argument & Human Validation Framework

> **목적**: 메타인지 평가 실험에서 사용하는 4+1 변환 타입(EA/SA/IC + TA)의
> unsolvability를 논문에서 정당화하기 위한 논증 구조와 human validation 프레임워크 정리.

---

## 1. 하드코딩 솔루션 발견

### 1.1 정량 분석

FinanceReasoning 데이터셋의 `python_solution` 필드를 분석한 결과,
대다수가 context를 프로그래밍적으로 파싱하지 않는 **하드코딩 스타일**임을 확인.

| 난이도 | 전체 문제 수 | 하드코딩 수 | 비율 |
|--------|-------------|------------|------|
| easy   | 1,000       | 979        | 97.9% |
| medium | 1,000       | 910        | 91.0% |
| hard   | 238         | 168        | 70.6% |
| **전체** | **2,238** | **2,057**  | **91.9%** |

**탐지 기준** (`_solution_uses_hardcoded_values()`):
context 파싱 패턴(`json.loads`, `.split(`, `context`, `for ... in`, `re.*(`, `.strip(`,
`.replace(`, `eval(`, `float('...`, `int('...`)이 하나도 없으면 하드코딩으로 판정.

### 1.2 솔루션 스타일 분류

데이터셋에서 관찰된 3가지 python_solution 스타일:

#### Style A: DataFrame 참조 (easy 중심)

```python
# test-0 (easy)
irn_gbp_2019 = df["GBP"]["FY 2019"]
answer = 1.0 / irn_gbp_2019
```

`df` 변수를 참조하지만, 키 이름(`"GBP"`, `"FY 2019"`)이 리터럴로 하드코딩됨.
context가 변환되어도 `df`가 실행 환경에 주입되지 않으면 동일하게 실패.

#### Style B: `def solution()` 함수 (hard 중심)

```python
# test-2002 (hard)
def solution():
    market_value_equity = 150000000  # 150 million
    market_value_debt = 100000000    # 100 million
    cost_of_equity = 0.09            # 9%
    cost_of_debt = 0.05              # 5%
    tax_rate = 0.30                  # 30%
    ...
```

context에서 추출한 값을 변수에 직접 할당. 함수 내부에서 context 참조 없음.

#### Style C: 직접 변수 할당

```python
# test-2000 (hard)
shares_outstanding = 1328
acquisition_cost = 176
shares_sold = 0
...
answer = shares_outstanding + acquisition_cost - shares_sold + ...
```

가장 단순한 형태. context를 전혀 읽지 않고 숫자를 그대로 대입.

### 1.3 난이도별 context 유형 분포

| 난이도 | JSON | Markdown | Text | None | 합계 |
|--------|------|----------|------|------|------|
| easy   | 189  | 570      | 212  | 29   | 1,000 |
| medium | 100  | 264      | 540  | 96   | 1,000 |
| hard   | 0    | 39       | 156  | 43   | 238 |

- hard에는 JSON context가 없음 → Type 1/2/4/5의 JSON 변환은 적용 불가
- hard의 65.5%가 text 형태

---

## 2. 자동 Validation의 한계

### 2.1 근본 문제

`validate_transformation()`은 ground-truth `python_solution`을 `exec()`으로 실행하여
변환 후에도 정답이 나오는지 확인하는 방식이지만, **91.9%의 솔루션이 context를 읽지 않아
변환과 무관하게 동일한 결과를 생성**:

```
[변환 전] context: "Revenue: $100M" → exec(python_solution) → answer = 42
[변환 후] context: "[DATA MISSING]"  → exec(python_solution) → answer = 42  (동일!)
```

따라서 자동 validation은 context-parsing 솔루션(~8.1%)에서만 유효.
하드코딩 솔루션의 경우 "still_solvable"로 판정되지만, 이는 **솔루션의 한계이지
변환의 실패가 아님**.

### 2.2 현재 필터링 전략

`hardcoded_solution_detector.py`에서 구현한 차등 처리:

| Phase B 응답 | 하드코딩 문제 | 처리 | 이유 |
|-------------|-------------|------|------|
| CONFIDENT + 정답 | Yes | **제외** | 변환 무효 — context 변경이 영향 없음 |
| CONFIDENT + 오답 | Yes | 유지 | 환각 — 유효한 데이터 |
| REFUSED | Yes | 유지 | context 이상 감지 — 유효한 메타인지 신호 |
| CAVEAT / ERROR | Yes | 유지 | 부분 인식 — 유효한 데이터 |

---

## 3. Design Argument: 변환별 Unsolvability 논증

### 3.1 이론적 프레임워크 (v2)

변환은 두 개의 직교 축으로 분류된다:

| 메타인지 능력 | 신호 강함 (Explicit) | 신호 없음 (Silent) |
|--------------|---------------------|-------------------|
| **부재 탐지** | EA-partial / EA-full | SA |
| **충돌 탐지** | — | IC |
| *(보조) 모호성 인식* | — | *TA* |

참고 문헌:
- AbstentionBench (Feng+2024): Underspecified Context vs Contradictory Data 분리
- Wen+2024 (EMNLP): context perturbation을 신호 강도로 분류
- CheckList (Ribeiro+2020): INV/DIR 테스트 — 변형 타입별 테스트 능력 분리

### 3.2 핵심 논거

**변환의 unsolvability는 python_solution의 실행 결과가 아니라,
변환의 설계 논리(design logic)에 의해 보장된다.**

각 변환은 문제 풀이에 *필수적인* 정보를 제거하거나 오염시키도록 설계되었으며,
이 설계 논리 자체가 unsolvability의 1차 근거이다.

### 3.3 변환 타입별 Unsolvability 논증

| Type | 변환 | 설계 논리 | 신뢰도 |
|------|------|-----------|--------|
| EA-partial | 부분 제거 + 명시적 마커 | question이 참조하는 키의 연도 값을 삭제하거나 `[DATA MISSING]`으로 대체. question이 직접 언급한 데이터가 없으므로 풀이 불가. Baseline — 가장 쉬운 탐지 과제. | **High** |
| EA-full | 키/컬럼 전체 삭제 | question이 참조하는 전체 데이터 카테고리를 삭제. 해당 차원의 정보가 완전히 소실. Markdown/JSON에서만 적용 (text는 구조 없음). | **High** |
| SA | 무표지 부재 | JSON: 값 비움 `{}`; Markdown: 데이터 행 삭제; Text: 핵심 문장 통째로 삭제. 마커 없이 맥락이 자연스러워 보이지만 풀이 필수 정보 부재. | **High** |
| IC | 정보 충돌 | 동일 데이터에 대해 1.5× 상이한 값을 추가. 두 값 중 어느 것이 correct인지 판단 근거 없음. 부재 탐지와 근본적으로 다른 인지 과정. | **Medium** — 모순을 탐지 못하면 기존 값으로 풀이 가능 |
| TA (보조) | 시간 참조 모호화 | question의 특정 연도를 "the end of the period"로 대체. ~11개 hard 문제에만 적용. 보조 분석으로 별도 보고. | **Medium** — 단일 연도 데이터 시 여전히 풀릴 가능성 |

### 3.4 변환 설계의 객관성

모든 변환은 다음 원칙을 따른다:

1. **결정론적(Deterministic)**: 랜덤 요소 없음. 동일 입력 → 동일 출력
2. **질문 기반 타겟팅(Question-driven)**: question 텍스트와 context 키를 매칭하여
   풀이에 *필수적인* 데이터만 타겟
3. **재현 가능(Reproducible)**: `apply_transformations()` 함수로 완전 재현

### 3.5 신뢰도 매트릭스

| 변환 타입 | Design 논증 | 자동 검증 가능 | Human Validation 필요성 |
|-----------|------------|---------------|----------------------|
| EA-partial | High | ~8%만 가능 | Low — 명시적 마커로 자명 |
| EA-full | High | ~8%만 가능 | Low — 키 삭제는 구조적으로 명확 |
| SA | High | ~8%만 가능 | Medium — 문장 삭제 범위 확인 필요 |
| IC | Medium | ~8%만 가능 | **High** — 모순의 실질적 영향 평가 |
| TA (보조) | Medium | ~8%만 가능 | **High** — 단일 연도 edge case |

---

## 4. Human Validation Framework

### 4.1 Transformation Precision 정의

$$
\text{Transformation Precision} = \frac{\text{truly unsolvable transformations}}{\text{total transformations sampled}}
$$

- **측정 단위**: 개별 (문제 × 변환 타입) 쌍
- **판정 기준**: 변환된 context/question만으로 ground_truth에 도달할 수 있는가?
- **판정자**: 금융 도메인 지식을 가진 인간 annotator

### 4.2 샘플링 전략

**층화 무작위 샘플링(Stratified Random Sampling)**:

| 층화 기준 | 수준 | 근거 |
|-----------|------|------|
| 난이도 | easy, medium, hard | 하드코딩 비율이 다름 |
| 변환 타입 | EA-partial, EA-full, SA, IC, TA | 타입별 신뢰도가 상이 |
| Context 유형 | json, markdown, text | 변환 함수가 다름 |

**목표 샘플 수**: ~120개 (통계적 유의성 확보)
- 5 타입 × 3 context 유형 = 최대 15 셀 (TA는 context 무관, EA-full은 text 제외)
- 셀당 ~8개 샘플 (가용 시, 불균등 분포 조정)

### 4.3 판정 체크리스트

각 샘플에 대해 annotator가 판정하는 항목:

| 항목 | 판정 | 설명 |
|------|------|------|
| **Unsolvable?** | Yes / No / Ambiguous | 변환된 정보만으로 정답 도출 가능 여부 |
| **Critical removal?** | Yes / No | 제거/변환된 데이터가 풀이에 필수적인가 |
| **Realistic?** | Yes / No | 실제 금융 데이터에서 이런 결손이 발생할 수 있는가 |
| **Detectable?** | Easy / Medium / Hard | 인간이 정보 부족을 인지하기 얼마나 쉬운가 |

### 4.4 보고 방식

논문에서의 보고 형식:

```
We conducted human validation on a stratified random sample of N=120
transformed problems (5 types × 3 context types × ~8 samples per cell).
Two annotators with financial domain knowledge independently judged
whether each transformation renders the problem unsolvable.

Results:
- Overall Transformation Precision: XX.X% (95% CI: [XX.X, XX.X])
- Inter-annotator agreement (Cohen's κ): 0.XX
- Per-type precision: Type 1 (XX%), Type 2 (XX%), ..., Type 5 (XX%)
```

---

## 5. 논문 Limitation 섹션 초안

### Limitations

> **Automated validation coverage.** 91.9% of FinanceReasoning's
> `python_solution` entries use hardcoded numeric values rather than
> programmatic context parsing, preventing automated verification of
> transformation unsolvability via solution execution. We address this
> through a two-pronged approach: (1) **design-based argumentation**
> demonstrating that each transformation type targets information
> structurally required by the question, and (2) **human validation**
> on a stratified sample of N=120 transformed problems, achieving
> a Transformation Precision of XX.X%.
>
> **Transformation coverage asymmetry.** The `hard` subset (our primary
> experimental target) contains no JSON-formatted contexts and 18.1%
> empty contexts (43/238), limiting the applicability of JSON-specific
> transformations (EA-partial, EA-full, SA, IC JSON variants). Text-based
> transformations provide the primary coverage for this difficulty level.
>
> **TA edge cases.** Temporal ambiguity transformations (TA)
> assume multi-year data in the context. Problems with single-year
> contexts may remain solvable after transformation, as "the end of
> the period" unambiguously resolves to the only available year.
> Only ~11 hard problems qualify; reported as auxiliary analysis.
> Human validation specifically targets this edge case.
>
> **IC detection baseline.** Contradiction insertion (IC)
> uses a deterministic 1.5× multiplier, which is domain-agnostic
> and may produce values outside realistic financial ranges for
> certain metrics (e.g., multiplying a 30% tax rate yields 45%).
> Future work could explore domain-constrained contradiction
> generation.

---

## 6. 논문 구조 내 배치 가이드

| 논문 섹션 | 포함 내용 | 이 문서 참조 |
|-----------|----------|-------------|
| §3 Methodology | 변환 설계 원칙, 4+1 타입 정의, 이론적 프레임워크 | §3.1, §3.3, §3.4 |
| §3.X Validation | Design Argument + Human Validation | §3, §4 |
| §4 Experiments | Transformation Precision 수치 | §4.4 |
| §5 Results | 하드코딩 발견 + 필터링 영향 | §1, §2.2 |
| §6 Limitations | 자동 검증 한계, 커버리지 비대칭 | §5 |
| Appendix | 솔루션 스타일 예시, 상세 validation 결과 | §1.2, §4.3 |
