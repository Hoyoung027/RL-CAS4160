import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

LOG_DIR = "data/hw7_RLHF-Hopper_Hopper-v5_04-06-2026_14-28-26"

ea = EventAccumulator(LOG_DIR)
ea.Reload()

eval_returns = ea.Scalars("Eval_AverageReturn")
steps  = [s.step for s in eval_returns]
values = [s.value for s in eval_returns]

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(steps, values, color="#2196F3", linewidth=1.8, label="Eval AverageReturn")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

ax.set_xlabel("Iteration")
ax.set_ylabel("Eval AverageReturn")
ax.set_title("Hopper Backflip — RLHF Training Curve")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(steps))

plt.tight_layout()
plt.savefig("hopper_eval_return.png", bbox_inches="tight")
print("Saved: hopper_eval_return.png")
