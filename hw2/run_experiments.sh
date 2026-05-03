#!/bin/bash
# HW2 전체 실험 실행 스크립트
#
# 사용법:
#   nohup bash run_experiments.sh > data/logs/all.log 2>&1 &
#   bash run_experiments.sh          # 전체 실험 실행
#   bash run_experiments.sh 1        # Experiment 1 (CartPole)만 실행
#   bash run_experiments.sh 2        # Experiment 2 (HalfCheetah)만 실행
#   bash run_experiments.sh 3        # Experiment 3 (HumanoidStandup)만 실행
#   bash run_experiments.sh 4        # Experiment 4 (Reacher/PPO)만 실행

set -e
cd "$(dirname "$0")"

run_exp1() {
    echo "=========================================="
    echo " Experiment 1: CartPole"
    echo "=========================================="

    echo "[1/8] cartpole (vanilla)"
    python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
        --video_log_freq 10 --exp_name cartpole

    echo "[2/8] cartpole_rtg (reward-to-go)"
    python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
        -rtg --video_log_freq 10 --exp_name cartpole_rtg

    echo "[3/8] cartpole_na (advantage normalization)"
    python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
        -na --video_log_freq 10 --exp_name cartpole_na

    echo "[4/8] cartpole_rtg_na (rtg + normalization)"
    python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
        -rtg -na --video_log_freq 10 --exp_name cartpole_rtg_na

    echo "[5/8] cartpole_lb (large batch, vanilla)"
    python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
        --video_log_freq 10 --exp_name cartpole_lb

    echo "[6/8] cartpole_lb_rtg (large batch, reward-to-go)"
    python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
        -rtg --video_log_freq 10 --exp_name cartpole_lb_rtg

    echo "[7/8] cartpole_lb_na (large batch, advantage normalization)"
    python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
        -na --video_log_freq 10 --exp_name cartpole_lb_na

    echo "[8/8] cartpole_lb_rtg_na (large batch, rtg + normalization)"
    python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
        -rtg -na --video_log_freq 10 --exp_name cartpole_lb_rtg_na

    echo "Experiment 1 완료!"
}

run_exp2() {
    echo "=========================================="
    echo " Experiment 2: HalfCheetah (Baseline)"
    echo "=========================================="

    echo "[1/3] cheetah (no baseline)"
    python cas4160/scripts/run_hw2.py --env_name HalfCheetah-v4 \
        -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
        --video_log_freq 10 --exp_name cheetah

    echo "[2/3] cheetah_baseline (with baseline, bgs=5)"
    python cas4160/scripts/run_hw2.py --env_name HalfCheetah-v4 \
        -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
        --use_baseline -blr 0.01 -bgs 5 \
        --video_log_freq 10 --exp_name cheetah_baseline

    echo "[3/3] cheetah_baseline_bgs1 (with baseline, bgs=1, ablation)"
    python cas4160/scripts/run_hw2.py --env_name HalfCheetah-v4 \
        -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
        --use_baseline -blr 0.01 -bgs 1 \
        --video_log_freq 10 --exp_name cheetah_baseline_bgs1

    echo "Experiment 2 완료!"
}

run_exp3() {
    echo "=========================================="
    echo " Experiment 3: HumanoidStandup (GAE)"
    echo "=========================================="

    echo "[1/3] HumanoidStandup lambda=0 (pure TD)"
    python cas4160/scripts/run_hw2.py \
        --env_name HumanoidStandup-v5 --ep_len 100 \
        --discount 0.99 -n 50 -l 3 -s 128 -b 2000 -lr 0.001 \
        --use_reward_to_go --use_baseline --gae_lambda 0 \
        --video_log_freq 10 --exp_name HumanoidStandup_lambda0

    echo "[2/3] HumanoidStandup lambda=0.95"
    python cas4160/scripts/run_hw2.py \
        --env_name HumanoidStandup-v5 --ep_len 100 \
        --discount 0.99 -n 50 -l 3 -s 128 -b 2000 -lr 0.001 \
        --use_reward_to_go --use_baseline --gae_lambda 0.95 \
        --video_log_freq 10 --exp_name HumanoidStandup_lambda0.95

    echo "[3/3] HumanoidStandup lambda=1 (Monte Carlo)"
    python cas4160/scripts/run_hw2.py \
        --env_name HumanoidStandup-v5 --ep_len 100 \
        --discount 0.99 -n 50 -l 3 -s 128 -b 2000 -lr 0.001 \
        --use_reward_to_go --use_baseline --gae_lambda 1 \
        --video_log_freq 10 --exp_name HumanoidStandup_lambda1

    echo "Experiment 3 완료!"
}

run_exp4() {
    echo "=========================================="
    echo " Experiment 4: Reacher (PPO)"
    echo "=========================================="

    echo "[1/2] reacher (vanilla PG baseline)"
    python cas4160/scripts/run_hw2.py \
        --env_name Reacher-v4 --ep_len 1000 \
        --discount 0.99 -n 100 -b 5000 -lr 0.003 \
        -na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
        --video_log_freq 10 --exp_name reacher

    echo "[2/2] reacher_ppo (PPO)"
    python cas4160/scripts/run_hw2.py \
        --env_name Reacher-v4 --ep_len 1000 \
        --discount 0.99 -n 100 -b 5000 -lr 0.003 \
        -na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
        --use_ppo --n_ppo_epochs 4 --n_ppo_minibatches 4 \
        --video_log_freq 10 --exp_name reacher_ppo

    echo "Experiment 4 완료!"
}

# 인자에 따라 실행
case "${1:-all}" in
    1) run_exp1 ;;
    2) run_exp2 ;;
    3) run_exp3 ;;
    4) run_exp4 ;;
    all)
        run_exp1
        run_exp2
        run_exp3
        run_exp4
        echo ""
        echo "=========================================="
        echo " 모든 실험 완료!"
        echo "=========================================="
        ;;
    *)
        echo "사용법: bash run_experiments.sh [1|2|3|4]"
        echo "  인자 없음 : 전체 실험 실행"
        echo "  1         : CartPole"
        echo "  2         : HalfCheetah"
        echo "  3         : HumanoidStandup (GAE)"
        echo "  4         : Reacher (PPO)"
        exit 1
        ;;
esac