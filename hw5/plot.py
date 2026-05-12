"""
HW5 CQL 시각화 스크립트 — Figure 4 (Overestimation curve)

사용법:
  python plot.py                    # figures/ 폴더에 저장
  python plot.py --data-dir <path>  # 데이터 디렉토리 지정 (기본값: data)
  python plot.py --out-dir <path>   # 출력 디렉토리 지정 (기본값: figures)
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


COLORS = ["tab:blue", "tab:orange"]


# ──────────────────────────────────────────────
# TensorBoard 파싱 (hw4 visualize.py 방식)
# ──────────────────────────────────────────────

def extract_scalars(log_dir, keys):
    if isinstance(keys, str):
        keys = [keys]
    scalars = {k: {"step": [], "value": []} for k in keys}
    tag_set = set(keys)

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
                if data_len > 8000:
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


def find_log_dir(data_dir, pattern):
    """data_dir 내에서 pattern을 포함하는 가장 최신 폴더를 반환"""
    matches = glob.glob(os.path.join(data_dir, f"*{pattern}*"))
    if not matches:
        return None
    return sorted(matches)[-1]  # 가장 최신 실행


def plot_line(ax, scalars, key, label, color):
    steps = scalars[key]["step"]
    values = scalars[key]["value"]
    if steps:
        ax.plot(steps, values, color=color, label=label)
    else:
        print(f"  [경고] '{key}' 데이터 없음 — {label}")


def format_steps(ax):
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k")
    )


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  저장: {path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Figure 4: Expert 데이터셋 Overestimation 커브
# ──────────────────────────────────────────────

def figure4(data_dir, out_dir):
    """Figure 4: Expert α=0.0 vs α=0.1 의 Overestimation value 커브"""
    log_a00 = find_log_dir(data_dir, "cql_alpha_0.0_expert")
    log_a01 = find_log_dir(data_dir, "cql_alpha_0.1_expert")

    if log_a00 is None:
        print("[Figure 4] cql_alpha_0.0_expert 폴더 없음, 건너뜀")
        return
    if log_a01 is None:
        print("[Figure 4] cql_alpha_0.1_expert 폴더 없음, 건너뜀")
        return

    print(f"  α=0.0 폴더: {log_a00}")
    print(f"  α=0.1 폴더: {log_a01}")

    key = "Overestimation"
    s00 = extract_scalars(log_a00, key)
    s01 = extract_scalars(log_a01, key)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_line(ax, s00, key, r"$\alpha=0.0$ (DQN)", COLORS[0])
    plot_line(ax, s01, key, r"$\alpha=0.1$ (CQL)", COLORS[1])

    format_steps(ax)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"Overestimation Value")
    ax.set_title("Overestimation of Q-values — Expert Dataset")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_fig(fig, os.path.join(out_dir, "figure4_overestimation.png"))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HW5 CQL 시각화")
    parser.add_argument("--data-dir", default="data", help="실험 데이터 루트 디렉토리")
    parser.add_argument("--out-dir",  default="figures", help="출력 PNG 디렉토리")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[오류] 데이터 디렉토리 없음: {args.data_dir}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    figure4(args.data_dir, args.out_dir)
    print("\n완료. 생성된 파일:", args.out_dir)


if __name__ == "__main__":
    main()
