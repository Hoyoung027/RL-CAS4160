from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import numpy as np
import os


def extract_tensorboard_scalars(log_dir, scalar_keys):
    import struct
    import glob
    from tensorboard.compat.proto.event_pb2 import Event

    if isinstance(scalar_keys, str):
        scalar_keys = [scalar_keys]

    scalars = {tag: {"step": [], "value": []} for tag in scalar_keys}
    tag_set = set(scalar_keys)

    # Scalar events are tiny (<1KB); video/image events are MBs — skip them via seek
    SKIP_THRESHOLD = 8000

    event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    for ef in event_files:
        with open(ef, "rb") as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                data_len = struct.unpack("<Q", header)[0]
                f.read(4)  # masked CRC of length

                if data_len > SKIP_THRESHOLD:
                    f.seek(data_len + 4, 1)  # skip data + CRC entirely
                    continue

                data = f.read(data_len)
                if len(data) < data_len:
                    break
                f.read(4)  # masked CRC of data

                event = Event()
                try:
                    event.ParseFromString(data)
                except Exception:
                    continue

                if not event.HasField("summary"):
                    continue
                for value in event.summary.value:
                    if value.tag in tag_set and value.HasField("simple_value"):
                        scalars[value.tag]["step"].append(event.step)
                        scalars[value.tag]["value"].append(value.simple_value)

    return scalars


def compute_mean_std(scalars: List[Dict[str, Any]], data_key: str, ninterp=100):
    min_step = min([s for slog in scalars for s in slog[data_key]["step"]])
    max_step = max([s for slog in scalars for s in slog[data_key]["step"]])
    steps = np.linspace(min_step, max_step, ninterp)
    scalars_interp = np.stack(
        [
            np.interp(
                steps,
                slog[data_key]["step"],
                slog[data_key]["value"],
                left=float("nan"),
                right=float("nan"),
            )
            for slog in scalars
        ],
        axis=1,
    )
    mean = np.nanmean(scalars_interp, axis=1)
    std = np.nanstd(scalars_interp, axis=1)
    return steps, mean, std


def plot_mean_std(ax, steps, mean, std, name, color):
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.3)
    ax.plot(steps, mean, color=color, label=name)


def plot_scalars(ax, scalars, data_key, name, color):
    ax.plot(scalars[data_key]["step"], scalars[data_key]["value"], color=color, label=name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_log_files", "-i", nargs="+", required=True)
    parser.add_argument("--human_readable_names", "-n", nargs="+", default=None)
    parser.add_argument("--colors", "-c", nargs="+", default=None)
    parser.add_argument("--data_key", "-d", nargs="+", type=str, required=True)
    parser.add_argument("--plot_mean_std", "-std", action="store_true")
    parser.add_argument("--plot_type", type=str, default="single", choices=["single", "subplot"],
                        help="single: one axes; subplot: one subplot per data_key")
    parser.add_argument("--title", "-t", type=str, default=None)
    parser.add_argument("--x_label_name", "-x", type=str, default="Environment Steps")
    parser.add_argument("--y_label_name", "-y", type=str, default=None)
    parser.add_argument("--output_file", "-o", type=str, default="output_plot.png")
    args = parser.parse_args()

    has_names = args.human_readable_names is not None

    if args.plot_type == "subplot":
        # One subplot per data_key; overlay all input_log_files on each subplot
        n_keys = len(args.data_key)
        fig, axes = plt.subplots(1, n_keys, figsize=(6 * n_keys, 4))
        if n_keys == 1:
            axes = [axes]

        colors = args.colors or [None] * len(args.input_log_files)
        names = args.human_readable_names or [None] * len(args.input_log_files)

        for ax, key in zip(axes, args.data_key):
            for log, name, color in zip(args.input_log_files, names, colors):
                scalars = extract_tensorboard_scalars(log, key)
                plot_scalars(ax, scalars, key, name, color)
            ax.set_xlabel(args.x_label_name or "Environment Steps")
            ax.set_ylabel(key)
            ax.set_title(key)
            if has_names:
                ax.legend()

        if args.title:
            fig.suptitle(args.title)
        plt.tight_layout()

    else:
        # single axes mode (hw2-compatible)
        if args.plot_mean_std:
            data_key = args.data_key[0]
            color = (args.colors or [None])[0]
            name = (args.human_readable_names or [None])[0]
            all_scalars = [extract_tensorboard_scalars(log, data_key) for log in args.input_log_files]
            xs, mean, std = compute_mean_std(all_scalars, data_key)
            plot_mean_std(plt.gca(), xs, mean, std, name, color)
        else:
            colors = args.colors or [None] * len(args.input_log_files)
            names = args.human_readable_names or [None] * len(args.input_log_files)
            for log, name, color in zip(args.input_log_files, names, colors):
                for key in args.data_key:
                    scalars = extract_tensorboard_scalars(log, key)
                    plot_name = f"{name}-{key}" if len(args.data_key) > 1 else name
                    plot_scalars(plt.gca(), scalars, key, plot_name, color)

        if has_names:
            plt.legend()
        if args.title:
            plt.title(args.title)
        if args.x_label_name:
            plt.xlabel(args.x_label_name)
        if args.y_label_name:
            plt.ylabel(args.y_label_name)

    plt.savefig(args.output_file, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output_file}")
    plt.show()
