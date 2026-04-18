"""
HW3 그래프 생성 스크립트

사용법:
  python cas4160/scripts/plot_hw3.py --figure 1 -i <log_dir> -o figures/figure1.png
  python cas4160/scripts/plot_hw3.py --figure 2 -i <default_lr_dir> <lr5e2_dir> -n "lr=0.001" "lr=0.05" -o figures/figure2.png
  python cas4160/scripts/plot_hw3.py --figure 3 --dqn-dirs <dqn_s1> <dqn_s2> <dqn_s3> --ddqn-dirs <ddqn_s1> <ddqn_s2> <ddqn_s3> -o figures/figure3.png
  python cas4160/scripts/plot_hw3.py --figure 4 -i <d0.5_dir> <d0.8_dir> <d0.99_dir> <d0.999_dir> -n "γ=0.5" "γ=0.8" "γ=0.99" "γ=0.999" -o figures/figure4.png
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from parse_tensorboard import extract_tensorboard_scalars, compute_mean_std, plot_mean_std, plot_scalars


COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def generate_figure1(log_dirs, output):
    """Figure 1: CartPole DQN eval_return 학습 곡선 (단일 실행)"""
    fig, ax = plt.subplots(figsize=(7, 4))
    scalars = extract_tensorboard_scalars(log_dirs[0], "eval_return")
    plot_scalars(ax, scalars, "eval_return", "DQN (lr=0.001)", COLORS[0])
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Return")
    ax.set_title("DQN on CartPole-v1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Figure 1 saved to {output}")


def generate_figure2(log_dirs, names, output):
    """Figure 2: 두 LR 비교 — q_values / critic_loss / eval_return 3개 subplot"""
    if len(log_dirs) != 2:
        raise ValueError("Figure 2 requires exactly 2 log directories (default lr, lr=0.05)")
    if names is None:
        names = ["lr=0.001 (default)", "lr=0.05"]

    keys = ["q_values", "critic_loss", "eval_return"]
    ylabels = ["Predicted Q-Values", "Critic Loss", "Eval Return"]
    titles = ["(a) Predicted Q-Values", "(b) Critic Error", "(c) Eval Return"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, key, ylabel, title in zip(axes, keys, ylabels, titles):
        for log, name, color in zip(log_dirs, names, COLORS):
            scalars = extract_tensorboard_scalars(log, key)
            plot_scalars(ax, scalars, key, name, color)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.suptitle("DQN on CartPole-v1: Learning Rate Comparison")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Figure 2 saved to {output}")


def generate_figure3(dqn_dirs, ddqn_dirs, output):
    """Figure 3: DQN vs Double DQN on BankHeist — mean±std (3 seeds each)"""
    if len(dqn_dirs) == 0 or len(ddqn_dirs) == 0:
        raise ValueError("Figure 3 requires --dqn-dirs and --ddqn-dirs")

    fig, ax = plt.subplots(figsize=(8, 5))

    dqn_scalars = [extract_tensorboard_scalars(d, "eval_return") for d in dqn_dirs]
    ddqn_scalars = [extract_tensorboard_scalars(d, "eval_return") for d in ddqn_dirs]

    xs_dqn, mean_dqn, std_dqn = compute_mean_std(dqn_scalars, "eval_return")
    xs_ddqn, mean_ddqn, std_ddqn = compute_mean_std(ddqn_scalars, "eval_return")

    plot_mean_std(ax, xs_dqn, mean_dqn, std_dqn, f"DQN (n={len(dqn_dirs)})", "tab:blue")
    plot_mean_std(ax, xs_ddqn, mean_ddqn, std_ddqn, f"Double DQN (n={len(ddqn_dirs)})", "tab:red")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Return")
    ax.set_title("DQN vs Double DQN on BankHeist-v5")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Figure 3 saved to {output}")


def generate_figure4(log_dirs, names, output):
    """Figure 4: 하이퍼파라미터(discount) 비교 — eval_return"""
    if names is None:
        names = [f"run_{i}" for i in range(len(log_dirs))]

    fig, ax = plt.subplots(figsize=(8, 5))

    for log, name, color in zip(log_dirs, names, COLORS):
        scalars = extract_tensorboard_scalars(log, "eval_return")
        plot_scalars(ax, scalars, "eval_return", name, color)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Return")
    ax.set_title("DQN on CartPole-v1: Discount Factor Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Figure 4 saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="HW3 그래프 생성")
    parser.add_argument("--figure", type=int, required=True, choices=[1, 2, 3, 4],
                        help="생성할 Figure 번호")
    parser.add_argument("-i", "--input_log_dirs", nargs="+", default=[],
                        help="로그 디렉토리 경로 (Figure 1, 2, 4용)")
    parser.add_argument("-n", "--names", nargs="+", default=None,
                        help="범례 이름 (Figure 2, 4용)")
    parser.add_argument("--dqn-dirs", nargs="+", default=[],
                        help="DQN 로그 디렉토리 (Figure 3용, 3개)")
    parser.add_argument("--ddqn-dirs", nargs="+", default=[],
                        help="Double DQN 로그 디렉토리 (Figure 3용, 3개)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="출력 PNG 파일 경로")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    output = args.output or f"figures/figure{args.figure}.png"

    if args.figure == 1:
        if not args.input_log_dirs:
            parser.error("Figure 1: -i <log_dir> 필요")
        generate_figure1(args.input_log_dirs, output)

    elif args.figure == 2:
        if len(args.input_log_dirs) != 2:
            parser.error("Figure 2: -i <default_lr_dir> <lr5e2_dir> 2개 필요")
        generate_figure2(args.input_log_dirs, args.names, output)

    elif args.figure == 3:
        generate_figure3(args.dqn_dirs, args.ddqn_dirs, output)

    elif args.figure == 4:
        if len(args.input_log_dirs) < 2:
            parser.error("Figure 4: -i <dir1> <dir2> ... 2개 이상 필요")
        generate_figure4(args.input_log_dirs, args.names, output)


if __name__ == "__main__":
    main()
