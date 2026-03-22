"""
DAgger 학습 곡선 플롯 (Eval_AverageReturn ± Eval_StdReturn)

Usage (from hw1/ directory):
    # 단일 로그
    python cas4160/scripts/plot_dagger.py \
        --log_dirs data/q2_dagger_ant_Ant-v4_...

    # 두 로그 동시에
    python cas4160/scripts/plot_dagger.py \
        --log_dirs data/q2_dagger_ant_Ant-v4_... data/q2_dagger_halfcheetah_... \
        --labels "Ant-v4" "HalfCheetah-v4"

    # BC/Expert 수평선 추가
    python cas4160/scripts/plot_dagger.py \
        --log_dirs data/q2_dagger_ant_Ant-v4_... \
        --bc_mean 1234.5 --expert_mean 4800.0
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from cas4160.scripts.parse_tensorboard import extract_tensorboard_scalars

COLORS = ["#2E8B57", "steelblue", "tomato", "mediumpurple"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dirs", "-i", nargs="+", required=True,
                        help="TensorBoard 로그 디렉터리 경로 (여러 개 가능)")
    parser.add_argument("--labels", "-n", nargs="+", default=None,
                        help="각 로그의 범례 이름 (생략 시 폴더명 사용)")
    parser.add_argument("--title", "-t", type=str, default="DAgger Learning Curve")
    parser.add_argument("--output_file", "-o", type=str, default="data/dagger_plot.png")
    parser.add_argument("--bc_mean", type=float, default=None,
                        help="BC 평균 return (수평선으로 표시)")
    parser.add_argument("--expert_mean", type=float, default=None,
                        help="Expert 평균 return (수평선으로 표시)")
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [os.path.basename(d.rstrip("/")) for d in args.log_dirs]

    fig, ax = plt.subplots(figsize=(10, 6))

    all_steps = []
    for log_dir, label, color in zip(args.log_dirs, args.labels, COLORS):
        scalars = extract_tensorboard_scalars(
            log_dir, ["Eval_AverageReturn", "Eval_StdReturn"]
        )
        steps = np.array(scalars["Eval_AverageReturn"]["step"])
        means = np.array(scalars["Eval_AverageReturn"]["value"])
        stds = np.array(scalars["Eval_StdReturn"]["value"])
        all_steps.extend(steps.tolist())

        print(f"\n[{label}]")
        print(f"  {'Step':>4}  {'Mean Return':>12}  {'Std':>10}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*10}")
        for s, m, d in zip(steps, means, stds):
            print(f"  {s:>4}  {m:>12.2f}  {d:>10.2f}")

        ax.errorbar(
            steps, means, yerr=stds,
            marker="o", linewidth=2, markersize=8,
            color=color, ecolor="dimgray", elinewidth=1.5,
            capsize=5, capthick=1.5,
            label=label,
        )

        for x, mean, std in zip(steps, means, stds):
            ax.annotate(
                f"{mean:.1f}±{std:.1f}",
                xy=(x, mean),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                color="dimgray",
            )

    if args.bc_mean is not None:
        ax.axhline(args.bc_mean, linestyle="--", color="orange", linewidth=1.5, label="BC")

    if args.expert_mean is not None:
        ax.axhline(args.expert_mean, linestyle="--", color="gray", linewidth=1.5, label="Expert")

    ax.set_ylim(3000, 5500)
    ax.set_xlabel("DAgger Iterations", fontsize=13)
    ax.set_ylabel("Mean Return", fontsize=13)
    ax.set_title(args.title, fontsize=14)
    ax.set_xticks(sorted(set(all_steps)))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=150)
    print(f"그래프 저장 완료 → {args.output_file}")


if __name__ == "__main__":
    main()
