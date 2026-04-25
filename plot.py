"""
plot.py — Training curve visualizations for DDPG on Pendulum-v1.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def smooth(values: list, window: int = 10) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def make_plots(
    rewards:     list,
    c_losses:    list,
    avg_rewards: list,
    save_path:   str = "results/training_curves.png",
    window:      int = 20,
) -> None:
    """
    Four-panel training diagnostics:
      1. Episode reward + moving average
      2. Critic MSE loss
      3. Cumulative average reward
      4. Reward distribution by training phase
    """
    ep_x = np.arange(1, len(rewards) + 1)

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = gridspec.GridSpec(2, 2, hspace=0.46, wspace=0.34)

    def ax_style(ax, title):
        ax.set_facecolor("#16213e")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#b0bec5", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#37474f")

    # ── Panel 1: reward trajectory ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax_style(ax1, "Episode Reward")
    ax1.plot(ep_x, rewards,     color="#4fc3f7", alpha=0.35, lw=0.9, label="Raw")
    ax1.plot(ep_x, avg_rewards, color="#ffd54f", lw=2.1,            label=f"{window}-ep avg")
    ax1.axhline(-200, color="#69f0ae", ls="--", lw=1.0, alpha=0.7, label="Optimal ≈ −200")
    ax1.set_xlabel("Episode",      color="#b0bec5", fontsize=9)
    ax1.set_ylabel("Total Reward", color="#b0bec5", fontsize=9)
    ax1.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#37474f", labelcolor="white")

    # ── Panel 2: critic loss ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax_style(ax2, "Critic (Q-Network) Loss")
    nz = [(i + 1, v) for i, v in enumerate(c_losses) if v > 0]
    if nz:
        ex, ey = zip(*nz)
        ax2.plot(ex, ey, color="#ef9a9a", alpha=0.40, lw=0.9, label="Per-episode")
        if len(ey) > 8:
            sm = smooth(list(ey), 8)
            ax2.plot(list(ex)[7:], sm, color="#ff5252", lw=2.1, label="8-ep smooth")
    ax2.set_xlabel("Episode", color="#b0bec5", fontsize=9)
    ax2.set_ylabel("MSE",     color="#b0bec5", fontsize=9)
    ax2.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#37474f", labelcolor="white")

    # ── Panel 3: cumulative average ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax_style(ax3, "Cumulative Average Reward")
    cum_avg = np.cumsum(rewards) / ep_x
    ax3.plot(ep_x, cum_avg, color="#a5d6a7", lw=2.1)
    ax3.fill_between(ep_x, cum_avg, alpha=0.12, color="#a5d6a7")
    ax3.set_xlabel("Episode",    color="#b0bec5", fontsize=9)
    ax3.set_ylabel("Avg Reward", color="#b0bec5", fontsize=9)

    # ── Panel 4: reward distribution by phase ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax_style(ax4, "Reward Distribution by Phase")
    t = len(rewards) // 3
    ax4.hist(rewards[:t],    bins=18, alpha=0.65, color="#ef9a9a", label="Early")
    ax4.hist(rewards[t:2*t], bins=18, alpha=0.65, color="#fff176", label="Mid")
    ax4.hist(rewards[2*t:],  bins=18, alpha=0.65, color="#80cbc4", label="Late")
    ax4.set_xlabel("Episode Reward", color="#b0bec5", fontsize=9)
    ax4.set_ylabel("Count",          color="#b0bec5", fontsize=9)
    ax4.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#37474f", labelcolor="white")

    plt.suptitle(
        "DDPG on Pendulum-v1  ·  Training Diagnostics",
        color="white", fontsize=13, fontweight="bold", y=0.99,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
