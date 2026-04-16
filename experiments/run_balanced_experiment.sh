#!/usr/bin/env bash
# =============================================================================
# Balanced 모델 실험 실행 스크립트
#
# 모델: gpt-4o, claude-sonnet-4, gemini-2.5-pro (balanced budget set)
# 입력: batch_transformations JSON (Phase 0에서 생성)
# 프롬프트 전략: metacognitive (기본), 추가 전략 순차 실행
#
# 주의: 실제 API 호출이 발생합니다. 비용을 확인하세요.
#       --dry-run 플래그로 설정만 검증 가능 (API 호출 없음)
#
# Usage:
#   ./experiments/run_balanced_experiment.sh              # dry-run (기본)
#   ./experiments/run_balanced_experiment.sh --execute     # 실제 실행
#   ./experiments/run_balanced_experiment.sh --strategy all # 모든 전략
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results/metacognitive"
INPUT_FILE="$RESULTS_DIR/batch_transformations_0_238.json"

# 기본값
DRY_RUN=true
STRATEGY="metacognitive"
CONCURRENCY=3

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --execute)
            DRY_RUN=false
            shift
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} Balanced 모델 실험 설정${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. 환경 확인
echo -e "${YELLOW}[1/4] 환경 확인${NC}"

# .env 파일 확인
if [ -f "$PROJECT_DIR/.env" ]; then
    echo -e "  ${GREEN}✓${NC} .env 파일 존재"
    # API 키 존재 여부만 확인 (값은 표시하지 않음)
    if grep -q "OPENAI_API_KEY" "$PROJECT_DIR/.env" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} OPENAI_API_KEY 설정됨"
    else
        echo -e "  ${RED}✗${NC} OPENAI_API_KEY 미설정 (gpt-4o 실행 불가)"
    fi
    if grep -q "ANTHROPIC_API_KEY" "$PROJECT_DIR/.env" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} ANTHROPIC_API_KEY 설정됨"
    else
        echo -e "  ${RED}✗${NC} ANTHROPIC_API_KEY 미설정 (claude-sonnet-4 실행 불가)"
    fi
    if grep -q "GOOGLE_API_KEY" "$PROJECT_DIR/.env" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} GOOGLE_API_KEY 설정됨"
    else
        echo -e "  ${RED}✗${NC} GOOGLE_API_KEY 미설정 (gemini-2.5-pro 실행 불가)"
    fi
else
    echo -e "  ${RED}✗${NC} .env 파일 없음 — API 키 설정 필요"
fi
echo ""

# 2. 입력 파일 확인
echo -e "${YELLOW}[2/4] 입력 파일 확인${NC}"
if [ -f "$INPUT_FILE" ]; then
    PROBLEM_COUNT=$(python3 -c "
import json
with open('$INPUT_FILE', encoding='utf-8') as f:
    data = json.load(f)
results = data.get('results', data) if isinstance(data, dict) else data
print(len(results) if isinstance(results, list) else len(results))
" 2>/dev/null || echo "?")
    echo -e "  ${GREEN}✓${NC} 입력: $INPUT_FILE"
    echo -e "  ${GREEN}✓${NC} 문제 수: $PROBLEM_COUNT"
else
    echo -e "  ${RED}✗${NC} 입력 파일 없음: $INPUT_FILE"
    echo -e "  ${YELLOW}→${NC} Phase 0 변환을 먼저 실행하세요:"
    echo "    python experiments/run_batch_transformation.py --start 0 --end 238"
fi
echo ""

# 3. 실행 계획 출력
echo -e "${YELLOW}[3/4] 실행 계획${NC}"
echo -e "  모델셋: ${GREEN}balanced${NC} (gpt-4o, claude-sonnet-4, gemini-2.5-pro)"
echo -e "  동시성: ${CONCURRENCY}"

# 전략 목록 구성
if [ "$STRATEGY" = "all" ]; then
    STRATEGIES=("metacognitive" "standard" "self_verification" "contradiction_aware")
else
    STRATEGIES=("$STRATEGY")
fi

echo -e "  프롬프트 전략:"
for s in "${STRATEGIES[@]}"; do
    echo -e "    - $s"
done

# 예상 비용 (balanced 모델 기준, 238문제)
echo ""
echo -e "  ${YELLOW}예상 비용 (238문제 기준):${NC}"
echo "    gpt-4o:          ~\$0.50-1.00/전략"
echo "    claude-sonnet-4:  ~\$0.30-0.60/전략"
echo "    gemini-2.5-pro:   ~\$0.10-0.20/전략"
echo "    전략당 합계:      ~\$0.90-1.80"
echo "    총 예상 비용:     ~\$$(echo "${#STRATEGIES[@]} * 1.35" | bc) (${#STRATEGIES[@]}개 전략)"
echo ""

# 4. 실행
echo -e "${YELLOW}[4/4] 실행${NC}"

if [ "$DRY_RUN" = true ]; then
    echo -e "  ${YELLOW}DRY-RUN 모드${NC} — 아래 명령어를 검토 후 --execute로 실행하세요:"
    echo ""
    for s in "${STRATEGIES[@]}"; do
        echo "  python experiments/run_batch_evaluation.py \\"
        echo "      --input $INPUT_FILE \\"
        echo "      --budget balanced \\"
        echo "      --prompt-strategy $s \\"
        echo "      --concurrency $CONCURRENCY"
        echo ""
    done
    echo -e "  ${BLUE}실행하려면:${NC} ./experiments/run_balanced_experiment.sh --execute"
else
    echo -e "  ${RED}실제 API 호출을 시작합니다!${NC}"
    read -p "  계속하시겠습니까? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "  취소됨."
        exit 0
    fi

    cd "$PROJECT_DIR"
    for s in "${STRATEGIES[@]}"; do
        echo ""
        echo -e "  ${GREEN}▶ 전략 실행: $s${NC}"
        python3 experiments/run_batch_evaluation.py \
            --input "$INPUT_FILE" \
            --budget balanced \
            --prompt-strategy "$s" \
            --concurrency "$CONCURRENCY"
        echo -e "  ${GREEN}✓ 완료: $s${NC}"
    done

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} 모든 실험 완료!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "결과: $RESULTS_DIR/"
fi
