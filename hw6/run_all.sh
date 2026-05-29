#!/bin/bash
cd /root/RL-CAS4160/hw6

# Section 4: model training only
echo "[1/5] Running 0_iter..."
python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_0_iter.yaml \
  > logs/exp_0iter.log 2>&1

# Section 5: random shooting
echo "[2/5] Running multi_iter (random shooting)..."
python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_multi_iter.yaml \
  > logs/exp_multi.log 2>&1

# Section 5: CEM x3 병렬
echo "[3-5/5] Running CEM experiments in parallel..."
python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_cem.yaml \
  > logs/exp_cem.log 2>&1 &
python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_cem_H-1.yaml \
  > logs/exp_cem_H1.log 2>&1 &
python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_cem_K-50.yaml \
  > logs/exp_cem_K50.log 2>&1 &

wait
echo "All experiments done!"
