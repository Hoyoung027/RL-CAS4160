"""
HW4 SAC 시각화 스크립트

사용법:
  python visualize.py                    # 모든 Figure 생성
  python visualize.py --figure 1         # Figure 1만 생성 (§6 InvPendulum REINFORCE)
  python visualize.py --figure 2         # Figure 2 (§6 HalfCheetah REINFORCE-1 vs 10)
  python visualize.py --figure 3         # Figure 3 (§7 HalfCheetah 3-way 비교)
  python visualize.py --figure 4         # Figure 4 (§8 Hopper Q-backup 비교)
  python visualize.py --data-dir <path>  # 데이터 디렉토리 지정 (기본값: data)
  python visualize.py --out-dir <path>   # 출력 디렉토리 지정 (기본값: figures)

폴더 번호 → 실험 매핑:
  1  : §4  sanity_pendulum_1         (Q값 안정화 확인)
  2  : §5  sanity_pendulum_2         (entropy 수렴 확인)
  3  : §6  sanity_invpendulum_reinforce
  4  : §6  halfcheetah_reinforce1
  5  : §6  halfcheetah_reinforce10
  6  : §7  sanity_invpendulum_reparametrize
  7  : §7  halfcheetah_reparametrize
  8  : §8  hopper_singleq
  9  : §8  hopper_doubleq
  10 : §8  hopper_clipq
"""

import argparse
import glob
import os
import struct
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event


COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

FOLDER_MAP = {
    1:  "sanity_pendulum_1",
    2:  "sanity_pendulum_2",
    3:  "sanity_invpendulum_reinforce",
    4:  "halfcheetah_reinforce1",
    5:  "halfcheetah_reinforce10",
    6:  "sanity_invpendulum_reparametrize",
    7:  "halfcheetah_reparametrize",
    8:  "hopper_singleq",
    9:  "hopper_doubleq",
    10: "hopper_clipq",
}


# ──────────────────────────────────────────────
# TensorBoard 파싱
# ──────────────────────────────────────────────

def extract_scalars(log_dir, keys):
    if isinstance(keys, str):
        keys = [keys]
    scalars = {k: {"step": [], "value": []} for k in keys}
    tag_set = set(keys)
    SKIP = 8000

    event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    if not event_files:
        print(f"  [경고] 이벤트 파일 없음: {log_dir}")
        return scalars

    for ef in event_files:
        with open(ef, "rb") as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                data_len = struct.unpack("<Q", header)[0]
                f.read(4)
                if data_len > SKIP:
                    f.seek(data_len + 4, 1)
                    continue
                data = f.read(data_len)
                if len(data) < data_len:
                    break
                f.read(4)
                event = Event()
                try:
                    event.ParseFromString(data)
                except Exception:
                    continue
                if not event.HasField("summary"):
                    continue
                for v in event.summary.value:
                    if v.tag in tag_set and v.HasField("simple_value"):
                        scalars[v.tag]["step"].append(event.step)
                        scalars[v.tag]["value"].append(v.simple_value)
    return scalars


def find_log_dir(data_dir, num):
    pattern = os.path.join(data_dir, f"{num}_*")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return sorted(matches)[0]


def plot_line(ax, scalars, key, label, color):
    steps = scalars[key]["step"]
    values = scalars[key]["value"]
    if steps:
        ax.plot(steps, values, color=color, label=label)
    else:
        print(f"  [경고] '{key}' 데이터 없음 — {label}")


def format_steps(ax):
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x < 1_000_000 else f"{x/1_000_000:.1f}M")
    )


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  저장: {path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Figure 생성 함수
# ──────────────────────────────────────────────

def figure1(data_dir, out_dir):
    """§6: InvertedPendulum REINFORCE — eval_return"""
    log = find_log_dir(data_dir, 3)
    if log is None:
        print("[Figure 1] 폴더 3 (sanity_invpendulum_reinforce) 없음, 건너뜀")
        return

    scalars = extract_scalars(log, "eval_return")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_line(ax, scalars, "eval_return", "REINFORCE", COLORS[0])
    ax.axhline(1000, color="gray", linestyle="--", linewidth=1, label="Target (1000)")
    format_steps(ax)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Return")
    ax.set_title("InvertedPendulum-v4 — REINFORCE")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "figure1_invpendulum_reinforce.png"))


def figure2(data_dir, out_dir):
    """§6: HalfCheetah REINFORCE-1 vs REINFORCE-10 — eval_return"""
    log1 = find_log_dir(data_dir, 4)
    log10 = find_log_dir(data_dir, 5)
    if log1 is None or log10 is None:
        print("[Figure 2] 폴더 4 또는 5 없음, 건너뜀")
        return

    s1  = extract_scalars(log1,  "eval_return")
    s10 = extract_scalars(log10, "eval_return")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_line(ax, s1,  "eval_return", "REINFORCE-1",  COLORS[0])
    plot_line(ax, s10, "eval_return", "REINFORCE-10", COLORS[1])
    ax.axhline(0,   color="gray", linestyle=":",  linewidth=1, label="Return = 0")
    ax.axhline(500, color="gray", linestyle="--", linewidth=1, label="Return = 500")
    format_steps(ax)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Return")
    ax.set_title("HalfCheetah-v4 — REINFORCE-1 vs REINFORCE-10")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "figure2_halfcheetah_reinforce.png"))


def figure3(data_dir, out_dir):
    """§7: HalfCheetah REINFORCE-1 / REINFORCE-10 / REPARAMETRIZE — eval_return"""
    log1     = find_log_dir(data_dir, 4)
    log10    = find_log_dir(data_dir, 5)
    log_rep  = find_log_dir(data_dir, 7)
    if None in (log1, log10, log_rep):
        print("[Figure 3] 폴더 4, 5, 7 중 일부 없음, 건너뜀")
        return

    s1    = extract_scalars(log1,    "eval_return")
    s10   = extract_scalars(log10,   "eval_return")
    s_rep = extract_scalars(log_rep, "eval_return")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_line(ax, s1,    "eval_return", "REINFORCE-1",    COLORS[0])
    plot_line(ax, s10,   "eval_return", "REINFORCE-10",   COLORS[1])
    plot_line(ax, s_rep, "eval_return", "REPARAMETRIZE",  COLORS[2])
    format_steps(ax)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Return")
    ax.set_title("HalfCheetah-v4 — Gradient Estimator Comparison")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "figure3_halfcheetah_comparison.png"))


def figure4(data_dir, out_dir):
    """§8: Hopper single-Q / double-Q / clip-Q — eval_return + q_values"""
    log_s = find_log_dir(data_dir, 8)
    log_d = find_log_dir(data_dir, 9)
    log_c = find_log_dir(data_dir, 10)
    if None in (log_s, log_d, log_c):
        print("[Figure 4] 폴더 8, 9, 10 중 일부 없음, 건너뜀")
        return

    keys = ["eval_return", "q_values"]
    ss = extract_scalars(log_s, keys)
    sd = extract_scalars(log_d, keys)
    sc = extract_scalars(log_c, keys)

    labels    = ["Single-Q", "Double-Q", "Clipped Double-Q"]
    scalars   = [ss, sd, sc]
    ylabels   = ["Eval Return", "Q Values"]
    titles    = ["(a) Eval Return", "(b) Q Values"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, ylabel, title in zip(axes, keys, ylabels, titles):
        for s, label, color in zip(scalars, labels, COLORS):
            plot_line(ax, s, key, label, color)
        format_steps(ax)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Hopper-v4 — Q-Backup Strategy Comparison (seed 48)")
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "figure4_hopper_qbackup.png"))


# ──────────────────────────────────────────────
# 보너스: §4 Q값 / §5 entropy 확인용
# ──────────────────────────────────────────────

def figure_sanity(data_dir, out_dir):
    """§4 Q값 안정화, §5 entropy 수렴 확인"""
    log4 = find_log_dir(data_dir, 1)
    log5 = find_log_dir(data_dir, 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if log4:
        s4 = extract_scalars(log4, "q_values")
        plot_line(axes[0], s4, "q_values", "Q values", COLORS[0])
        axes[0].axhline(-1000, color="gray", linestyle="--", linewidth=1, label="Expected (−1000)")
    format_steps(axes[0])
    axes[0].set_xlabel("Environment Steps")
    axes[0].set_ylabel("Q Values")
    axes[0].set_title("Sec 4: Pendulum-v1 — Q-value Stabilization")
    axes[0].legend()

    if log5:
        s5 = extract_scalars(log5, "entropy")
        plot_line(axes[1], s5, "entropy", "Entropy", COLORS[1])
        axes[1].axhline(0.693, color="gray", linestyle="--", linewidth=1, label="log 2 ≈ 0.693")
    format_steps(axes[1])
    axes[1].set_xlabel("Environment Steps")
    axes[1].set_ylabel("Entropy")
    axes[1].set_title("Sec 5: Pendulum-v1 — Entropy Convergence")
    axes[1].legend()

    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "figure0_sanity.png"))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HW4 SAC 시각화")
    parser.add_argument("--figure", type=int, choices=[0, 1, 2, 3, 4],
                        help="생성할 Figure 번호 (생략 시 전체 생성)\n"
                             "  0: §4/§5 sanity check\n"
                             "  1: §6 InvPendulum REINFORCE\n"
                             "  2: §6 HalfCheetah REINFORCE-1 vs 10\n"
                             "  3: §7 HalfCheetah 3-way 비교\n"
                             "  4: §8 Hopper Q-backup 비교")
    parser.add_argument("--data-dir", default="data", help="실험 데이터 루트 디렉토리")
    parser.add_argument("--out-dir",  default="figures", help="출력 PNG 디렉토리")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir  = args.out_dir

    if not os.path.isdir(data_dir):
        print(f"[오류] 데이터 디렉토리 없음: {data_dir}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    generators = {
        0: figure_sanity,
        1: figure1,
        2: figure2,
        3: figure3,
        4: figure4,
    }

    if args.figure is not None:
        generators[args.figure](data_dir, out_dir)
    else:
        for fn in generators.values():
            fn(data_dir, out_dir)

    print("\n완료. 생성된 파일:", out_dir)


if __name__ == "__main__":
    main()
