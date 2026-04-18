#!/bin/bash
# 실행 방법 (백그라운드):
#   nohup bash run_all_experiments.sh &
#
# 로그 확인:
#   tail -f run_all_experiments_*.log

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/run_all_experiments_$(date '+%Y%m%d_%H%M%S').log"

# 터미널과 로그 파일 동시에 출력
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"

log() {
    echo ""
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "=========================================="
}

# Section 4.2 — CartPole
log "Section 4.2: CartPole (default LR)"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/cartpole.yaml --seed 1 -nvid 1

log "Section 4.2: CartPole (LR=5e-2)"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/cartpole_lr_5e-2.yaml --seed 1 -nvid 1

# Section 5.2 — BankHeist DQN (3 seeds)
log "Section 5.2: BankHeist DQN (seed 1)"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 1 -nvid 1

log "Section 5.2: BankHeist DQN (seed 2)"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 2 -nvid 1

log "Section 5.2: BankHeist DQN (seed 3)"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 3 -nvid 1

# Section 5.2 — BankHeist Double DQN (3 seeds)
log "Section 5.2: BankHeist Double DQN (seed 1)"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 1 -nvid 1

log "Section 5.2: BankHeist Double DQN (seed 2)"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 2 -nvid 1

log "Section 5.2: BankHeist Double DQN (seed 3)"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 3 -nvid 1

# Section 6 — Discount 하이퍼파라미터 실험
log "Section 6: CartPole discount=0.5"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/cartpole_discount_0.5.yaml --seed 1 -nvid 1

log "Section 6: CartPole discount=0.8"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/cartpole_discount_0.8.yaml --seed 1 -nvid 1

log "Section 6: CartPole discount=0.99"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/cartpole_discount_0.99.yaml --seed 1 -nvid 1

log "Section 6: CartPole discount=0.999"
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/cartpole_discount_0.999.yaml --seed 1 -nvid 1

log "All experiments done!"
