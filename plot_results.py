"""
plot_results.py
---------------
Generates all paper-ready figures for the GSQ experiments.

Figure 1: Training curves (loss + accuracy, Phase I vs II shading)
Figure 2: D_FS vs Delta scatter — GSQ vs Euclidean (Experiment A core result)
Figure 3: Ablation — K anchor count vs deployment shock
Figure 4: Ablation — tau delay fraction vs test accuracy

Usage:
    python plot_results.py --save_dir results --fig_dir figures
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import GSQConfig

# ── Guard flag for placeholder ablation data ──
# TODO: PLACEHOLDER — set to False BEFORE paper submission and provide real data!
USE_PLACEHOLDER_DATA = True

# IEEE-style plot settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "ideal":     "#2c7bb6",
    "euclidean": "#d7191c",
    "gsq":       "#1a9641",
    "phase1":    "#1ff7bc",
    "phase2":    "#49f0a3",
}


# ─────────────────────────────────────────────
# Figure 1: Training curves
# ─────────────────────────────────────────────

def plot_training_curves(history: dict, tau: float = 0.6,
                          save_path: str = None):
    epochs = len(history["loss_total"])
    t0 = int(tau * epochs)
    xs = np.arange(1, epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    for ax in axes:
        ax.axvspan(1, t0, alpha=0.15, color=COLORS["phase1"],
                   label="Phase I (free exploration)")
        ax.axvspan(t0, epochs, alpha=0.15, color=COLORS["phase2"],
                   label="Phase II (geometric hardening)")
        ax.axvline(t0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Loss
    axes[0].plot(xs, history["loss_task"],  color=COLORS["ideal"],
                 lw=1.5, label="Task loss")
    axes[0].plot(xs, history["loss_geom"],  color=COLORS["euclidean"],
                 lw=1.2, linestyle="--", label="Geom. regularizer")
    axes[0].plot(xs, history["loss_total"], color="#333333",
                 lw=1.8, label="Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("(a) Training loss")
    axes[0].legend(fontsize=8, loc="upper right")

    # Accuracy
    axes[1].plot(xs, history["train_acc"], color=COLORS["ideal"],
                 lw=1.5, label="Train acc.")
    axes[1].plot(xs, history["test_acc"],  color=COLORS["gsq"],
                 lw=1.5, linestyle="--", label="Test acc.")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("(b) Classification accuracy")

    p1 = mpatches.Patch(color=COLORS["phase1"], alpha=0.5, label="Phase I")
    p2 = mpatches.Patch(color=COLORS["phase2"], alpha=0.5, label="Phase II")
    axes[1].legend(handles=[
        plt.Line2D([0], [0], color=COLORS["ideal"], lw=1.5, label="Train"),
        plt.Line2D([0], [0], color=COLORS["gsq"], lw=1.5, linestyle="--", label="Test"),
        p1, p2
    ], fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Figure 2: D_FS vs Delta (core result)
# ─────────────────────────────────────────────

def plot_dfs_vs_delta(results: dict, save_path: str = None):
    """
    Bar chart comparing D_FS and Delta for Euclidean vs GSQ.
    This is the central 'geometry matters' figure.
    """
    methods = ["Euclidean", "GSQ (ours)"]
    colors  = [COLORS["euclidean"], COLORS["gsq"]]
    D_FS    = [results["euclidean"]["D_FS"],  results["gsq"]["D_FS"]]
    delta   = [results["euclidean"]["delta"], results["gsq"]["delta"]]

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))

    x = np.arange(len(methods))
    w = 0.5

    axes[0].bar(x, D_FS, width=w, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel(r"$D_{\mathrm{FS}}$ (geometric distortion)")
    axes[0].set_title(r"(a) Fubini–Study distortion $D_{\mathrm{FS}}$")
    for i, v in enumerate(D_FS):
        axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold")

    axes[1].bar(x, delta, width=w, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel(r"$\Delta$ (deployment shock)")
    axes[1].set_title(r"(b) Operational deployment shock $\Delta$")
    for i, v in enumerate(delta):
        axes[1].text(i, v + 0.0005, f"{v:.4f}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold")

    red_pct  = results["improvement"]["D_FS_reduction"] * 100
    shk_pct  = results["improvement"]["shock_reduction"] * 100
    fig.suptitle(
        f"GSQ reduces $D_{{\\mathrm{{FS}}}}$ by {red_pct:.1f}% "
        f"and $\\Delta$ by {shk_pct:.1f}% vs. Euclidean quantization",
        fontsize=9, y=1.02
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Figure 3: K ablation
# ─────────────────────────────────────────────

def plot_k_ablation(k_values: list, shock_gsq: list, shock_euclid: list,
                    save_path: str = None):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(k_values, shock_gsq, "o-", color=COLORS["gsq"],
            lw=1.8, markersize=5, label="GSQ (circular)")
    ax.plot(k_values, shock_euclid, "s--", color=COLORS["euclidean"],
            lw=1.5, markersize=5, label="Euclidean K-means")
    ax.set_xlabel("Number of anchors $K$")
    ax.set_ylabel(r"Deployment shock $\Delta$")
    ax.set_title("(c) Effect of anchor count on deployment shock")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Figure 4: tau ablation
# ─────────────────────────────────────────────

def plot_tau_ablation(tau_values: list, acc_test: list, shock: list,
                      save_path: str = None):
    fig, ax1 = plt.subplots(figsize=(4.5, 3))
    ax2 = ax1.twinx()
    ax1.plot(tau_values, acc_test, "o-", color=COLORS["ideal"],
             lw=1.8, markersize=5, label="Test accuracy")
    ax2.plot(tau_values, shock, "s--", color=COLORS["euclidean"],
             lw=1.5, markersize=5, label=r"Shock $\Delta$")
    ax1.set_xlabel(r"Delay fraction $\tau$")
    ax1.set_ylabel("Test accuracy", color=COLORS["ideal"])
    ax2.set_ylabel(r"Deployment shock $\Delta$", color=COLORS["euclidean"])
    ax1.set_title(r"(d) Effect of geometric hardening delay $\tau$")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(save_dir: str = "results", fig_dir: str = "figures"):
    os.makedirs(fig_dir, exist_ok=True)

    # Load training history
    hist_path = os.path.join(save_dir, "train_history.json")
    with open(hist_path) as f:
        history = json.load(f)

    # Read tau from saved config or checkpoint instead of hardcoding
    config_path = os.path.join(save_dir, "config.json")
    if os.path.exists(config_path):
        cfg = GSQConfig.load(config_path)
        tau = cfg.tau
    else:
        # Fallback: try to read from checkpoint hyperparams
        weights_path = os.path.join(save_dir, "trained_weights.pt")
        if os.path.exists(weights_path):
            import torch
            ckpt = torch.load(weights_path, weights_only=True)
            tau = ckpt.get("hyperparams", {}).get("tau", 0.6)
        else:
            tau = 0.6  # last-resort default

    # Load deployment results
    dep_path = os.path.join(save_dir, "deployment_results.json")
    with open(dep_path) as f:
        results = json.load(f)

    print("\n  Generating figures...\n")

    plot_training_curves(
        history, tau=tau,
        save_path=os.path.join(fig_dir, "fig1_training_curves.pdf")
    )
    plot_dfs_vs_delta(
        results,
        save_path=os.path.join(fig_dir, "fig2_dfs_delta.pdf")
    )

    # ── Ablation figures ──────────────────────────────────────────
    # TODO: PLACEHOLDER DATA — replace with real ablation sweep results
    # before paper submission! Run ablation sweeps and save results to
    # e.g. results/k_ablation.json and results/tau_ablation.json,
    # then load here instead of using these fabricated arrays.
    if USE_PLACEHOLDER_DATA:
        import warnings
        warnings.warn(
            "⚠️  Ablation figures use PLACEHOLDER data! "
            "Set USE_PLACEHOLDER_DATA = False and load real sweep results "
            "before submitting the paper.",
            UserWarning,
            stacklevel=2,
        )

        k_vals  = [2, 4, 6, 8, 10, 12, 16]
        np.random.seed(0)
        shock_g = np.array([0.18, 0.12, 0.09, 0.07, 0.06, 0.055, 0.05]) \
                  + np.random.normal(0, 0.003, 7)
        shock_e = np.array([0.22, 0.17, 0.14, 0.12, 0.11, 0.105, 0.10]) \
                  + np.random.normal(0, 0.003, 7)
        plot_k_ablation(k_vals, shock_g.tolist(), shock_e.tolist(),
                        save_path=os.path.join(fig_dir, "fig3_k_ablation.pdf"))

        tau_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        acc_t = np.array([0.62, 0.66, 0.71, 0.75, 0.79, 0.82, 0.81, 0.78, 0.72])
        shk_t = np.array([0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.075, 0.09, 0.11])
        plot_tau_ablation(tau_vals, acc_t.tolist(), shk_t.tolist(),
                          save_path=os.path.join(fig_dir, "fig4_tau_ablation.pdf"))
    else:
        # Load real ablation sweep data
        k_abl_path = os.path.join(save_dir, "k_ablation.json")
        tau_abl_path = os.path.join(save_dir, "tau_ablation.json")

        if os.path.exists(k_abl_path):
            with open(k_abl_path) as f:
                k_data = json.load(f)
            plot_k_ablation(
                k_data["k_values"], k_data["shock_gsq"], k_data["shock_euclid"],
                save_path=os.path.join(fig_dir, "fig3_k_ablation.pdf")
            )
        else:
            print(f"  WARNING: {k_abl_path} not found, skipping K ablation plot.")

        if os.path.exists(tau_abl_path):
            with open(tau_abl_path) as f:
                tau_data = json.load(f)
            plot_tau_ablation(
                tau_data["tau_values"], tau_data["acc_test"], tau_data["shock"],
                save_path=os.path.join(fig_dir, "fig4_tau_ablation.pdf")
            )
        else:
            print(f"  WARNING: {tau_abl_path} not found, skipping tau ablation plot.")

    print("\n  All figures saved to:", fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--fig_dir",  type=str, default="figures")
    args = parser.parse_args()
    main(args.save_dir, args.fig_dir)
