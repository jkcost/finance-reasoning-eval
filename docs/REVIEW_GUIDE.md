# Human Review 협업 가이드

## 개요

이 리뷰의 목적은 **LLM 메타인지 평가를 위한 고품질 변환 데이터셋**을 만드는 것입니다.

원본 금융 문제(hard.json, 238문제)에서 풀이에 필요한 데이터를 자동으로 제거/변형하여
"unsolvable" 문제를 만들었습니다. 이 자동 변환이 올바른지(실제로 문제를 풀 수 없게 만드는지)
사람이 검토하는 것이 현재 작업입니다.

---

## 환경 설정

```bash
git clone https://github.com/jkcost/finance-reasoning-eval.git
cd finance-reasoning-eval
git checkout feat/metacognitive-evaluation-framework

# Python 환경 (conda 또는 venv)
pip install -r evaluation/requirements.txt
```

---

## 작업 흐름 (전체)

```
[1] 변환 생성          →  [2] HTML 생성         →  [3] 브라우저에서 리뷰
(한번만, 이미 완료)       (한번만)                  (각자 담당 범위)
                                                        ↓
[6] HTML 재생성        ←  [5] 머지 + 불일치해결  ←  [4] JSON 내보내기 → Git 커밋
(다음 라운드)              (관리자)                   (각자)
```

---

## Step 1~2: 변환 생성 + HTML 생성

처음 한번만 실행하면 됩니다. 이미 실행되어 있다면 건너뛰세요.

```bash
# 변환 생성 (API 호출 없음, 즉시 완료)
python experiments/run_batch_transformation.py --start 0 --end 30

# 리뷰 HTML 생성
python experiments/generate_human_review.py

# 결과 파일 확인
ls experiments/results/metacognitive/human_review_0_30.html
```

---

## Step 3: 브라우저에서 리뷰

### 담당 범위

| 작업자 | 범위 | 브라우저 URL |
|--------|------|-------------|
| 작업자 A | #0 ~ #9 | `human_review_0_30.html?assignee=작업자A&start=0&end=10` |
| 작업자 B | #10 ~ #19 | `human_review_0_30.html?assignee=작업자B&start=10&end=20` |
| 작업자 C | #20 ~ #29 | `human_review_0_30.html?assignee=작업자C&start=20&end=30` |

> `assignee`에 본인 이름을 입력하세요. 이름별로 작업 내용이 별도 저장됩니다.

### 리뷰 방법

1. **HTML 상단의 "변환 유형 가이드 & 리뷰 기준"을 먼저 펼쳐서 읽어주세요**
2. 문제 카드를 클릭하여 펼치기
3. **Python Solution** 섹션 확인:
   - `● 초록색` = 풀이에 사용되는 변수 (이 값이 제거되어야 올바른 변환)
   - `○ 회색` = 사용되지 않는 변수
4. 변환 탭(EA-partial, EA-full, SA, IC, TA)을 클릭하여 원본 vs 변환 비교
5. 판정 선택:

| 판정 | 기준 | 예시 |
|------|------|------|
| **승인** | 제거된 정보가 풀이에 필수, 남은 정보로 답 도출 불가 | ROE 계산에 필요한 순이익을 삭제 |
| **수정 필요** | 방향은 맞지만 제거 대상이나 방식에 개선 필요 | 풀이와 무관한 값을 삭제함 |
| **부적절** | 변환 후에도 문제를 풀 수 있음 | 삭제한 컬럼을 다른 컬럼으로 역산 가능 |

6. 수정이 필요한 경우 → `편집` 버튼으로 변환된 context를 직접 수정 가능

### 판정 시 체크 리스트

- [ ] 제거/변형된 데이터가 풀이에 **필수적인가?** (Python Solution의 ● 변수 확인)
- [ ] 남은 데이터로 답을 **역산할 수 있는가?**
      예: `A = B + C`이면 `B`를 삭제해도 `A - C`로 구할 수 있음 → 부적절
- [ ] (SA) 변환 후 context가 **자연스러운가?** 삭제 흔적이 보이면 수정 필요
- [ ] (IC) 모순 값이 **풀이에 사용되는 값인가?** 무관한 값에 모순을 넣으면 의미 없음

### 키보드 단축키

| 키 | 동작 |
|----|------|
| `J` / `K` | 다음 / 이전 문제 |
| `1` ~ `5` | EA-partial ~ TA 탭 전환 |
| `A` | 현재 탭 승인 |
| `R` | 현재 탭 부적절 |

---

## Step 4: JSON 내보내기 + Git 커밋

리뷰 완료 후:

1. HTML 상단의 **`JSON 내보내기`** 버튼 클릭
2. 다운로드된 파일을 아래 경로에 저장:
   ```
   experiments/results/metacognitive/annotations/review_본인이름.json
   ```
3. Git 커밋 & 푸시:
   ```bash
   cd finance-reasoning-eval
   mkdir -p experiments/results/metacognitive/annotations
   # 다운로드한 JSON 파일을 위 경로로 이동

   git add experiments/results/metacognitive/annotations/review_본인이름.json
   git commit -m "[review] 본인이름 annotation #0-#9"
   git push origin feat/metacognitive-evaluation-framework
   ```

> **주의**: `experiments/results/`는 gitignore에 있지만,
> `annotations/` 하위 JSON은 `git add`로 **강제 추가** 가능합니다 (`-f` 플래그 필요 시 사용).

---

## Step 5: 결과 통합 (관리자)

모든 작업자가 커밋한 후:

```bash
git pull

# 작업자 JSON 머지
python experiments/merge_annotations.py

# 결과:
#   annotations/merged.json       — 통합 결과
#   annotations/disagreements.json — 판정 불일치 목록 (있을 경우)
```

불일치 항목은 팀이 함께 검토하여 최종 판정을 결정합니다.

### 다음 라운드 HTML 재생성

```bash
python experiments/generate_human_review.py \
  --annotations experiments/results/metacognitive/annotations/merged.json
```

이전 리뷰 결과가 미리 채워진 HTML이 생성됩니다.

---

## 변환 유형 상세

### EA-partial (Explicit Absence — Partial)
- **방법**: 특정 셀/값을 `N/A`로 교체
- **탐지 난이도**: LOW — 마커가 명시적으로 보임
- **체크**: N/A로 바뀐 값이 풀이에 실제로 필요한지

### EA-full (Explicit Absence — Full)
- **방법**: 테이블 컬럼 또는 JSON 키 전체 삭제
- **탐지 난이도**: MODERATE — 구조적 변화는 있지만 마커 없음
- **체크**: 삭제된 컬럼의 값을 다른 컬럼으로 역산할 수 있는지 반드시 확인

### SA (Silent Absence)
- **방법**: 마커 없이 조용히 삭제 (문장/행/값 제거, 남은 문맥이 자연스러움)
- **탐지 난이도**: HIGH — 아무 단서 없음
- **체크**: 삭제 후 문맥이 자연스러운지, 삭제된 정보 없이 정말 풀 수 없는지

### IC (Information Conflict)
- **방법**: 원래 값의 1.5배인 모순 데이터를 삽입
- **탐지 난이도**: VERY HIGH — 현재 모든 모델이 탐지 실패
- **체크**: 모순이 자연스럽게 삽입되었는지, 모순 대상이 풀이에 사용되는 값인지

### TA (Temporal Ambiguity) [보조]
- **방법**: question의 연도를 "the end of the period"로 교체
- **적용**: ~11개 문제에만 적용 가능
- **체크**: context에 여러 연도 데이터가 있어 실제로 모호해지는지

---

## FAQ

**Q: 브라우저를 닫으면 작업이 사라지나요?**
A: 아닙니다. localStorage에 자동 저장됩니다. 같은 브라우저 + 같은 assignee로 열면 복원됩니다.
   단, 다른 컴퓨터에서는 보이지 않으므로 반드시 JSON 내보내기를 해주세요.

**Q: 변환이 실패한 탭(회색)도 리뷰해야 하나요?**
A: 아닙니다. 성공한 변환(초록색 테두리 탭)만 리뷰하면 됩니다.

**Q: context를 직접 편집하면 원본이 바뀌나요?**
A: 아닙니다. 편집 내용은 annotation JSON에만 저장됩니다.

**Q: 같은 문제를 두 명이 리뷰하면 어떻게 되나요?**
A: 판정이 일치하면 자동 통합, 불일치하면 disagreements.json에 기록됩니다.
