"""
Hyperparameter sweep over --n_layers for BC on Ant-v4.

Usage (from hw1/ directory):
    python cas4160/scripts/run_nlayers_sweep.py
    python cas4160/scripts/run_nlayers_sweep.py --plot_only  # JSON 이미 있을 때 그래프만
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

# parse_tensorboard.py 와 같은 패키지 안에 있으므로 직접 import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from cas4160.scripts.parse_tensorboard import extract_tensorboard_scalars

# ── 실험 설정 ──────────────────────────────────────────────
N_LAYERS_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
ENV_NAME = "Ant-v4"
EXPERT_POLICY = "cas4160/policies/experts/Ant.pkl"
EXPERT_DATA = "cas4160/expert_data/expert_data_Ant-v4.pkl"
EVAL_BATCH_SIZE = 10000

# data/ 폴더 위치 (run_hw1.py 와 동일한 기준)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
RESULTS_JSON = os.path.join(DATA_DIR, "nlayers_sweep_results.json")
PLOT_FILE = os.path.join(DATA_DIR, "nlayers_sweep_plot.png")


def find_log_dir(exp_name: str) -> str | None:
    """실험 이름으로 생성된 tensorboard 로그 디렉터리를 찾는다."""
    pattern = os.path.join(DATA_DIR, f"q1_{exp_name}_{ENV_NAME}_*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getctime)  # 가장 최근 생성된 것


def run_experiment(n_layers: int) -> str | None:
    exp_name = f"bc_ant_nlayers{n_layers}"
    cmd = [
        "python", "cas4160/scripts/run_hw1.py",
        "--expert_policy_file", EXPERT_POLICY,
        "--env_name", ENV_NAME,
        "--exp_name", exp_name,
        "--n_iter", "1",
        "--expert_data", EXPERT_DATA,
        "--eval_batch_size", str(EVAL_BATCH_SIZE),
        "--n_layers", str(n_layers),
    ]
    print(f"\n{'='*55}")
    print(f"  n_layers = {n_layers}")
    print(f"{'='*55}")
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    subprocess.run(cmd, check=True, env=env)
    return find_log_dir(exp_name)


def parse_log(log_dir: str) -> tuple[float, float]:
    """tensorboard 로그에서 Eval_AverageReturn, Eval_StdReturn 을 읽는다."""
    scalars = extract_tensorboard_scalars(
        log_dir, ["Eval_AverageReturn", "Eval_StdReturn"]
    )
    mean = scalars["Eval_AverageReturn"]["value"][0]
    std = scalars["Eval_StdReturn"]["value"][0]
    return mean, std


def run_sweep() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}

    for n_layers in N_LAYERS_LIST:
        log_dir = run_experiment(n_layers)
        if log_dir is None:
            print(f"[경고] n_layers={n_layers} 로그 디렉터리를 찾을 수 없습니다.")
            continue

        mean, std = parse_log(log_dir)
        results[str(n_layers)] = {"mean": mean, "std": std, "log_dir": log_dir}
        print(f"  결과: {mean:.2f} ± {std:.2f}")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장 완료 → {RESULTS_JSON}")
    return results


def plot_results(results: dict):
    n_vals = sorted(int(k) for k in results)
    means = [results[str(k)]["mean"] for k in n_vals]
    stds = [results[str(k)]["std"] for k in n_vals]

    color = "#2E8B57"  # sea green

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(n_vals, means, marker="o", linewidth=2, markersize=8, color=color, label="BC policy")

    # std를 각 점 옆에 텍스트로 표시
    for x, mean, std in zip(n_vals, means, stds):
        ax.annotate(
            f"±{std:.1f}",
            xy=(x, mean),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=11,
            color="dimgray",
        )

    ax.set_xlabel("Number of Hidden Layers (--n_layers)", fontsize=13)
    ax.set_ylabel("Eval Average Return", fontsize=13)
    ax.set_title("BC Performance vs. Policy Network Depth (Ant-v4)", fontsize=14)
    ax.set_xticks(n_vals)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"그래프 저장 완료 → {PLOT_FILE}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot_only", action="store_true",
        help="기존 JSON 결과로 그래프만 그린다 (실험 재실행 없음)"
    )
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(RESULTS_JSON):
            print(f"[오류] {RESULTS_JSON} 파일이 없습니다. 먼저 실험을 실행하세요.")
            sys.exit(1)
        with open(RESULTS_JSON) as f:
            results = json.load(f)
        print(f"JSON 로드 완료 ({len(results)}개 실험)")
    else:
        results = run_sweep()

    plot_results(results)


if __name__ == "__main__":
    main()