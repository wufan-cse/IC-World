import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import MinMaxScaler


def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size) / window_size, mode='valid')

def load_and_process(path, window_size, limit=100):
    data = np.loadtxt(path)
    data = moving_average(data[:limit], window_size)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return data

# Paths
paths = [
    "./outputs/grpo_geo_bs1_lr1e-5_gas8_ng32_lora_rank64_max101/reward.txt",
]
labels = ["reward"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# Load all
datasets = [load_and_process(p, window_size=20, limit=100) for p in paths]

fig, ax = plt.subplots(figsize=(4.5, 4.5))

for data, label, color in zip(datasets, labels, colors):
    x = np.arange(len(data))
    ax.plot(x, data, color=color, linewidth=2.2, label=label)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.set_xlabel("Training Steps", fontsize=18, labelpad=1, weight="bold")
ax.set_ylabel("Reward", fontsize=18, labelpad=1, weight="bold")
ax.tick_params(axis="x", labelsize=12, pad=1.5)
ax.tick_params(axis="y", labelsize=12, pad=1.5)

ax.legend(fontsize=13, loc="best", frameon=False)

plt.tight_layout(pad=0.3)
plt.savefig("reward.png", dpi=300, bbox_inches="tight")
plt.show()
