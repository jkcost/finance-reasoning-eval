# Local Model Experiment — Deferred Backlog

**상태**: DEFERRED (2026-04-21)
**재개 조건**: CIKM Short 본 실험(판정 이유 카테고리화 + Hard Example + balanced 3모델) 결과가 안정화된 후
**분리 이유**: 로컬 모델 실험은 GPU 인프라 의존성이 커서 본 프로젝트와 별도 트랙으로 진행하는 것이 합리적. 현재 스프린트(W2~W3)는 판정 이유·Hard Example에 집중.

## 배경

4/15 회의에서 최동희 교수가 제안: 1million 암기 같은 학습 편향이 오픈 모델에서도 재현되는지 보기 위해 API 모델(GPT/Claude/Gemini)에 더해 로컬 모델(Llama 등)을 HTML에 나란히 표시. 진수(Speaker 3)가 GPU 실행을 지원하겠다고 합의.

## 현재까지 준비된 것

| 항목 | 위치 | 상태 |
|------|------|------|
| OllamaProvider 구현 | `evaluation/model_runner.py` | done (커밋 739d902) |
| Ollama 모델 registry 등록 | `evaluation/error_analysis/error_taxonomy.py` | done — llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5:7b |
| `BUDGET_MODEL_SETS["local"]`, `["local-small"]` | 위와 동일 | done |
| `run_local_model_eval.py` | `experiments/` | done — 443 lines, Phase A/B 전용 |
| Ollama API 키 체크 skip | `init_providers()` | done |

## 재개 시 해야 할 일 (원래 W2 prep 계획)

### 6-P1: 진수에게 공식 요청
- **무엇을**: GPU 사양 확인 (A100? L40S? RTX 3090?) + 가용 일정 + 실행 장소 (로컬/연구실 공용)
- **채널**: Slack DM 또는 이메일
- **메시지 템플릿 (재개 시 사용)**:
  > 진규님 안녕하세요. 4/15 회의에서 논의한 로컬 모델 실험 진행하려 합니다.
  > 1) GPU 사양 알려주실 수 있을까요? (llama3.1:8b 돌릴 수 있으면 충분)
  > 2) 다음 주 중 실행 가능한 시간대
  > 3) 입력 파일 `experiments/results/metacognitive/batch_transformations_0_238.json` 드리면 됩니다. 출력은 API 모델과 동일 포맷(`evaluation_results_*.json`)으로 반환해 주시면 머지하겠습니다.

### 6-P2: 모델 후보 확정
- **현재 registry에 있음**: llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5:7b
- **추가 고려**: deepseek-r1 (reasoning 특화), qwen2.5-coder (POT 친화적)
- **결정 기준**: RAM 요구량 × 진수 GPU 스펙

### 6-P3: 로컬 dry-run (진수 기다리지 않고)
- 개발자 본인 Mac에서 Ollama 설치 → 1문제만 돌려서 포맷 확인
- 명령: `ollama pull llama3.1:8b && python experiments/run_local_model_eval.py --limit 1`

### 6-P4: 프롬프트 호환성 스팟 체크
- POT + metacognitive 프롬프트가 7~8B 모델에서 파싱 실패(`INSUFFICIENT_INFORMATION` 포맷 오인식) 빈도 측정
- 10문제 샘플 → 파싱 성공률 ≥80% 확인. 미달 시 fallback 프롬프트 설계

### 6-P5: 출력 머지 인터페이스
- 진수 결과 파일 형식을 API 모델과 통일
- 스키마: `{model, prompt_strategy, transformation_type, qid, response, refused, refusal_reason, ...}`
- 머지 스크립트: `experiments/merge_local_api_results.py` (구현 필요 시 추가)

## 재개 후 산출물 예상

- HTML 리뷰: API 3모델 + 로컬 2~3모델 나란히 비교
- 1million 암기 재현 분석: 오픈 모델에서도 재현되는지 / 모델 크기 의존성

## 관련 파일

- `evaluation/config.py` — LOCAL_MODELS 설정
- `evaluation/model_runner.py` — OllamaProvider
- `experiments/run_local_model_eval.py` — 로컬 전용 실행 스크립트

## 관련 문서

- `docs/meetings/2026-04-15-team-sync.md` Action Item #3
- `docs/memory/project_pot_faithfulness_rqb.md` — 1million 현상 (로컬 모델 재현 필요성)
