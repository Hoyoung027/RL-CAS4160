#!/bin/bash
set -e

export MUJOCO_GL=egl

HW7_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HW7_DIR/logs"
mkdir -p "$(pwd)/logs"

cd "$HW7_DIR/cas4160/scripts"

echo "================================================================"
echo "[1/3] PointMaze - synthetic preferences (자동, human feedback 없음)"
echo "================================================================"
python run_hw7.py \
    --env_name PointMaze_UMazeDense-v3 \
    --ep_len 150 --discount 0.99 -lr 0.003 -n 21 -b 2000 \
    --use_reward_to_go -na --use_baseline --gae_lambda 0.97 \
    --use_ppo --n_ppo_epochs 4 --n_ppo_minibatches 4 \
    --exp_name RLHF-PointMaze-Syn \
    --video_log_freq 4 --annotate_freq 2 \
    --init_annotate_step 8 --annotate_step 4 \
    --syn_prefs
echo "[1/3] 완료"

echo "================================================================"
echo "[2/3] PointMaze - human preferences"
echo "  --> 브라우저에서 :5000 접속해서 feedback 제공 필요"
echo "================================================================"
python run_hw7.py \
    --env_name PointMaze_UMazeDense-v3 \
    --ep_len 150 --discount 0.99 -lr 0.003 -n 21 -b 2000 \
    --use_reward_to_go -na --use_baseline --gae_lambda 0.97 \
    --use_ppo --n_ppo_epochs 4 --n_ppo_minibatches 4 \
    --exp_name RLHF-PointMaze-Human \
    --video_log_freq 4 --annotate_freq 2 \
    --init_annotate_step 8 --annotate_step 4
echo "[2/3] 완료"

echo "================================================================"
echo "[3/3] Hopper Backflip - human preferences"
echo "  --> 브라우저에서 :5000 접속해서 feedback 제공 필요 (~80분)"
echo "================================================================"
python run_hw7.py \
    --env_name Hopper-v5 \
    --ep_len 150 --discount 0.99 -lr 0.003 -n 100 -b 4000 \
    --use_reward_to_go -na --use_baseline --gae_lambda 0.97 \
    --use_ppo --n_ppo_epochs 4 --n_ppo_minibatches 4 \
    --exp_name RLHF-Hopper \
    --video_log_freq 5 --annotate_freq 5 \
    --init_annotate_step 48 --annotate_step 8
echo "[3/3] 완료"

echo "================================================================"
echo "전체 실험 완료"
echo "================================================================"
