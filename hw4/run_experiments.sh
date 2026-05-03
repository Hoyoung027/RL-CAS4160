#!/bin/bash
# HW4 SAC 실험 실행 스크립트
# 사용법:
#   전체 실행:          bash run_experiments.sh
#   특정 섹션만 실행:   bash run_experiments.sh 4
#                       bash run_experiments.sh 5
#                       bash run_experiments.sh 6
#                       bash run_experiments.sh 7
#                       bash run_experiments.sh 8
#   중단 후 재시작:     bash run_experiments.sh remaining

set -e

SCRIPT="/usr/local/bin/anaconda3/envs/cas4160/bin/python cas4160/scripts/run_hw4.py"
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local cfg=$1
    local extra=${2:-""}
    local name=$(basename "$cfg" .yaml)
    echo ""
    echo "========================================"
    echo "  실행: $name $extra"
    echo "  시작: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    $SCRIPT -cfg "$cfg" $extra 2>&1 | tee "$LOG_DIR/${name}.log"
    echo "  완료: $(date '+%Y-%m-%d %H:%M:%S')"
}

section=${1:-"all"}

# ──────────────────────────────────────────────
# Section 4: Bootstrapping (Critic only, ~수 분)
# 확인: Q값이 -700 ~ -1000 근처에서 안정화
# ──────────────────────────────────────────────
if [[ "$section" == "all" || "$section" == "4" ]]; then
    echo ""
    echo "### Section 4: Bootstrapping ###"
    run_exp experiments/sac/sanity_pendulum_1.yaml
fi

# ──────────────────────────────────────────────
# Section 5: Entropy Bonus (~수 분)
# 확인: entropy ≈ 0.69 (log 2) 수렴
# ──────────────────────────────────────────────
if [[ "$section" == "all" || "$section" == "5" ]]; then
    echo ""
    echo "### Section 5: Entropy Bonus ###"
    run_exp experiments/sac/sanity_pendulum_2.yaml
fi

# ──────────────────────────────────────────────
# Section 6: REINFORCE Actor Update (각 1시간+)
# 확인: InvertedPendulum return ≈ 1000
#       HalfCheetah REINFORCE-1: 500K 내 양수 reward
#       HalfCheetah REINFORCE-10: 200K 내 return > 500
# ──────────────────────────────────────────────
if [[ "$section" == "all" || "$section" == "6" ]]; then
    echo ""
    echo "### Section 6: REINFORCE ###"
    run_exp experiments/sac/sanity_invertedpendulum_reinforce.yaml
    run_exp experiments/sac/halfcheetah_reinforce1.yaml
    run_exp experiments/sac/halfcheetah_reinforce10.yaml
fi

if [[ "$section" == "remaining" ]]; then
    echo ""
    echo "### 중단된 실험 재시작 (halfcheetah_reinforce10 ~ hopper_clipq) ###"
    run_exp experiments/sac/halfcheetah_reinforce10.yaml
fi

# ──────────────────────────────────────────────
# Section 7: REPARAMETRIZE Actor Update (각 1시간+)
# 확인: InvertedPendulum return ≈ 1000
# ──────────────────────────────────────────────
if [[ "$section" == "all" || "$section" == "7" || "$section" == "remaining" ]]; then
    echo ""
    echo "### Section 7: REPARAMETRIZE ###"
    run_exp experiments/sac/sanity_invertedpendulum_reparametrize.yaml
    run_exp experiments/sac/halfcheetah_reparametrize.yaml
fi

# ──────────────────────────────────────────────
# Section 8: Stabilizing Target Values (각 20분)
# 확인: single-Q / double-Q / clipped-Q 비교
#       eval_return + q_values 플롯
# ──────────────────────────────────────────────
if [[ "$section" == "all" || "$section" == "8" || "$section" == "remaining" ]]; then
    echo ""
    echo "### Section 8: Q-backup Strategies ###"
    run_exp experiments/sac/hopper.yaml "--seed 48"
    run_exp experiments/sac/hopper_doubleq.yaml "--seed 48"
    run_exp experiments/sac/hopper_clipq.yaml "--seed 48"
fi

echo ""
echo "========================================"
echo "  모든 실험 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  로그 위치: $LOG_DIR/"
echo "========================================"
