"""
plot_results.py
---------------
Generates all 3-way comparative publication-ready figures for the GSQ experiments.

Figure 1: Training curves (2x2 grid, Two Moons and MNIST training details with Phase I/II shading)
Figure 2: Fubini-Study distortion and Deployment shock (1x2 grouped bar chart with error caps: m0 vs. m1 vs. m2)
Figure 3: Ablation — K anchor count vs deployment shock (1x2 subplots: m0 vs. m1 vs. m2)
Figure 4: Ablation — tau delay fraction vs test accuracy/shock (1x2 double-y subplots: m2 vs. m0 flat baselines)

Usage:
    python plot_results.py --save_dir results --fig_dir figures
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Disable placeholder data — GSQ now uses 100% real experimental sweeps!
USE_PLACEHOLDER_DATA = False

# IEEE-style plot settings for top-tier journals
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "m0":        "#777777",  # Standard VQC (Grey)
    "ideal":     "#2c7bb6",
    "euclidean": "#fdae61",  # Euclidean K-Means (Orange)
    "gsq":       "#1a9641",  # GSQ (Ours, Green)
    "phase1":    "#e0f3f8",
    "phase2":    "#e5f5e0",
}


# ─────────────────────────────────────────────
# Figure 1: Training curves (2x2 Grid)
# ─────────────────────────────────────────────

def plot_training_curves_multi(
    history_tm: dict,
    history_mn: dict,
    tau_tm: float = 0.6,
    tau_mn: float = 0.6,
    save_path: str = None
):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))

    # --- ROW 0: Two Moons ---
    epochs_tm = len(history_tm["loss_total"])
    t0_tm = int(tau_tm * epochs_tm)
    xs_tm = np.arange(1, epochs_tm + 1)

    # Shading Phases
    for col in range(2):
        ax = axes[0, col]
        ax.axvspan(1, t0_tm, alpha=0.6, color=COLORS["phase1"])
        ax.axvspan(t0_tm, epochs_tm, alpha=0.6, color=COLORS["phase2"])
        ax.axvline(t0_tm, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Loss
    axes[0, 0].plot(xs_tm, history_tm["loss_task"],  color=COLORS["ideal"], lw=1.2, label="Task loss")
    axes[0, 0].plot(xs_tm, history_tm["loss_geom"],  color="#d7191c", lw=1.0, linestyle="--", label="Geom. reg.")
    axes[0, 0].plot(xs_tm, history_tm["loss_total"], color="#333333", lw=1.5, label="Total loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("(a) Two Moons: Loss curves")
    axes[0, 0].legend(fontsize=7, loc="upper right")

    # Accuracy
    axes[0, 1].plot(xs_tm, history_tm["train_acc"], color=COLORS["ideal"], lw=1.2, label="Train")
    axes[0, 1].plot(xs_tm, history_tm["test_acc"],  color=COLORS["gsq"], lw=1.2, linestyle="--", label="Test")
    axes[0, 1].set_ylim(0.4, 1.05)
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("(b) Two Moons: Classification accuracy")
    
    p1 = mpatches.Patch(color=COLORS["phase1"], alpha=0.8, label="Phase I (explore)")
    p2 = mpatches.Patch(color=COLORS["phase2"], alpha=0.8, label="Phase II (hardening)")
    axes[0, 1].legend(handles=[
        plt.Line2D([0], [0], color=COLORS["ideal"], lw=1.2, label="Train"),
        plt.Line2D([0], [0], color=COLORS["gsq"], lw=1.2, linestyle="--", label="Test"),
        p1, p2
    ], fontsize=7, loc="lower right")

    # --- ROW 1: MNIST ---
    epochs_mn = len(history_mn["loss_total"])
    t0_mn = int(tau_mn * epochs_mn)
    xs_mn = np.arange(1, epochs_mn + 1)

    # Shading Phases
    for col in range(2):
        ax = axes[1, col]
        ax.axvspan(1, t0_mn, alpha=0.6, color=COLORS["phase1"])
        ax.axvspan(t0_mn, epochs_mn, alpha=0.6, color=COLORS["phase2"])
        ax.axvline(t0_mn, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Loss
    axes[1, 0].plot(xs_mn, history_mn["loss_task"],  color=COLORS["ideal"], lw=1.2)
    axes[1, 0].plot(xs_mn, history_mn["loss_geom"],  color="#d7191c", lw=1.0, linestyle="--")
    axes[1, 0].plot(xs_mn, history_mn["loss_total"], color="#333333", lw=1.5)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("(c) MNIST 0 vs 1: Loss curves")

    # Accuracy
    axes[1, 1].plot(xs_mn, history_mn["train_acc"], color=COLORS["ideal"], lw=1.2)
    axes[1, 1].plot(xs_mn, history_mn["test_acc"],  color=COLORS["gsq"], lw=1.2, linestyle="--")
    axes[1, 1].set_ylim(0.4, 1.05)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("(d) MNIST 0 vs 1: Classification accuracy")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        png_path = save_path.replace(".pdf", ".png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {save_path} and {png_path}")
    plt.close()


# ─────────────────────────────────────────────
# Figure 2: Grouped 3-Way Bar Chart with Error Caps
# ─────────────────────────────────────────────

def plot_dfs_vs_delta_multi(
    stat_tm: dict,
    stat_mn: dict,
    save_path: str = None
):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    datasets_lbl = ["Two Moons", "MNIST 0 vs 1"]
    
    # Extract means
    dfs_m0     = [stat_tm["metrics"]["m0_dfs"]["mean"],        stat_mn["metrics"]["m0_dfs"]["mean"]]
    dfs_euclid = [stat_tm["metrics"]["euclidean_dfs"]["mean"], stat_mn["metrics"]["euclidean_dfs"]["mean"]]
    dfs_gsq    = [stat_tm["metrics"]["gsq_dfs"]["mean"],      stat_mn["metrics"]["gsq_dfs"]["mean"]]
    
    shk_m0     = [stat_tm["metrics"]["m0_shock"]["mean"],        stat_mn["metrics"]["m0_shock"]["mean"]]
    shk_euclid = [stat_tm["metrics"]["euclidean_shock"]["mean"], stat_mn["metrics"]["euclidean_shock"]["mean"]]
    shk_gsq    = [stat_tm["metrics"]["gsq_shock"]["mean"],      stat_mn["metrics"]["gsq_shock"]["mean"]]

    # Extract standard deviations
    dfs_m0_std     = [stat_tm["metrics"]["m0_dfs"]["std"],        stat_mn["metrics"]["m0_dfs"]["std"]]
    dfs_euclid_std = [stat_tm["metrics"]["euclidean_dfs"]["std"], stat_mn["metrics"]["euclidean_dfs"]["std"]]
    dfs_gsq_std    = [stat_tm["metrics"]["gsq_dfs"]["std"],      stat_mn["metrics"]["gsq_dfs"]["std"]]
    
    shk_m0_std     = [stat_tm["metrics"]["m0_shock"]["std"],        stat_mn["metrics"]["m0_shock"]["std"]]
    shk_euclid_std = [stat_tm["metrics"]["euclidean_shock"]["std"], stat_mn["metrics"]["euclidean_shock"]["std"]]
    shk_gsq_std    = [stat_tm["metrics"]["gsq_shock"]["std"],      stat_mn["metrics"]["gsq_shock"]["std"]]

    x = np.arange(len(datasets_lbl))
    w = 0.22  # Bar width

    # (a) Fubini-Study Distortion Subplot (m0 vs m1 vs m2)
    axes[0].bar(x - w, dfs_m0,     yerr=dfs_m0_std,     width=w, color=COLORS["m0"],        edgecolor="none", capsize=2, error_kw={"lw": 0.6}, label="Standard VQC (m0)")
    axes[0].bar(x,     dfs_euclid, yerr=dfs_euclid_std, width=w, color=COLORS["euclidean"], edgecolor="none", capsize=2, error_kw={"lw": 0.6}, label="Euclidean VQC (m1)")
    axes[0].bar(x + w, dfs_gsq,    yerr=dfs_gsq_std,    width=w, color=COLORS["gsq"],       edgecolor="none", capsize=2, error_kw={"lw": 0.6}, label="GSQ Proposed (m2)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets_lbl)
    axes[0].set_ylabel(r"$D_{\mathrm{FS}}$ (FS distortion)")
    axes[0].set_title(r"(a) Fubini–Study distortion $D_{\mathrm{FS}}$")
    axes[0].legend(fontsize=7, loc="upper right")

    # Add text labels on top of bars
    for idx, (m0_val, e_val, g_val) in enumerate(zip(dfs_m0, dfs_euclid, dfs_gsq)):
        axes[0].text(idx - w, m0_val + dfs_m0_std[idx] + 0.0005, f"{m0_val:.4f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
        axes[0].text(idx,     e_val + dfs_euclid_std[idx] + 0.0005, f"{e_val:.4f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
        axes[0].text(idx + w, g_val + dfs_gsq_std[idx] + 0.0005, f"{g_val:.4f}", ha="center", va="bottom", fontsize=6, fontweight="bold")

    # (b) Deployment Shock Subplot (m0 vs m1 vs m2)
    axes[1].bar(x - w, shk_m0,     yerr=shk_m0_std,     width=w, color=COLORS["m0"],        edgecolor="none", capsize=2, error_kw={"lw": 0.6})
    axes[1].bar(x,     shk_euclid, yerr=shk_euclid_std, width=w, color=COLORS["euclidean"], edgecolor="none", capsize=2, error_kw={"lw": 0.6})
    axes[1].bar(x + w, shk_gsq,    yerr=shk_gsq_std,    width=w, color=COLORS["gsq"],       edgecolor="none", capsize=2, error_kw={"lw": 0.6})
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets_lbl)
    axes[1].set_ylabel(r"$\Delta$ (deployment shock)")
    axes[1].set_title(r"(b) Operational deployment shock $\Delta$")

    for idx, (m0_val, e_val, g_val) in enumerate(zip(shk_m0, shk_euclid, shk_gsq)):
        axes[1].text(idx - w, m0_val + shk_m0_std[idx] + 0.0001, f"{m0_val:.4f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
        axes[1].text(idx,     e_val + shk_euclid_std[idx] + 0.0001, f"{e_val:.4f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
        axes[1].text(idx + w, g_val + shk_gsq_std[idx] + 0.0001, f"{g_val:.4f}", ha="center", va="bottom", fontsize=6, fontweight="bold")

    # Extract significance testing results (GSQ m2 vs Standard m0)
    p_dfs_tm = stat_tm["statistical_tests"]["dfs"]["p_value"]
    d_dfs_tm = stat_tm["statistical_tests"]["dfs"]["cohen_d"]
    p_dfs_mn = stat_mn["statistical_tests"]["dfs"]["p_value"]
    d_dfs_mn = stat_mn["statistical_tests"]["dfs"]["cohen_d"]

    fig.suptitle(
        f"GSQ (m2) vs Standard VQC (m0) Significance: Two Moons (p={p_dfs_tm:.4f}, d={d_dfs_tm:.2f}) | "
        f"MNIST (p={p_dfs_mn:.4f}, d={d_dfs_mn:.2f})",
        fontsize=8, y=1.02
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        png_path = save_path.replace(".pdf", ".png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {save_path} and {png_path}")
    plt.close()


# ─────────────────────────────────────────────
# Figure 3: K Ablation (1x2 Subplots, 3 Curves)
# ─────────────────────────────────────────────

def plot_k_ablation_multi(
    k_data_tm: dict,
    k_data_mn: dict,
    save_path: str = None
):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # (a) Two Moons
    axes[0].plot(k_data_tm["k_values"], k_data_tm["shock_m0"], "x:", color=COLORS["m0"],
                 lw=1.2, markersize=4.0, label="Standard VQC (m0)")
    axes[0].plot(k_data_tm["k_values"], k_data_tm["shock_euclid"], "s--", color=COLORS["euclidean"],
                 lw=1.2, markersize=4.0, label="Euclidean VQC (m1)")
    axes[0].plot(k_data_tm["k_values"], k_data_tm["shock_gsq"], "o-", color=COLORS["gsq"],
                 lw=1.5, markersize=4.5, label="GSQ Circular (m2)")
    axes[0].set_xlabel("Number of anchors $K$")
    axes[0].set_ylabel(r"Deployment shock $\Delta$")
    axes[0].set_title("(a) Two Moons: Anchor ablation")
    axes[0].legend(fontsize=7)

    # (b) MNIST
    axes[1].plot(k_data_mn["k_values"], k_data_mn["shock_m0"], "x:", color=COLORS["m0"],
                 lw=1.2, markersize=4.0)
    axes[1].plot(k_data_mn["k_values"], k_data_mn["shock_euclid"], "s--", color=COLORS["euclidean"],
                 lw=1.2, markersize=4.0)
    axes[1].plot(k_data_mn["k_values"], k_data_mn["shock_gsq"], "o-", color=COLORS["gsq"],
                 lw=1.5, markersize=4.5)
    axes[1].set_xlabel("Number of anchors $K$")
    axes[1].set_ylabel(r"Deployment shock $\Delta$")
    axes[1].set_title("(b) MNIST 0 vs 1: Anchor ablation")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        png_path = save_path.replace(".pdf", ".png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {save_path} and {png_path}")
    plt.close()


# ─────────────────────────────────────────────
# Figure 4: tau Ablation (1x2 Double-Y Subplots, with m0 baseline)
# ─────────────────────────────────────────────

def plot_tau_ablation_multi(
    tau_data_tm: dict,
    tau_data_mn: dict,
    save_path: str = None
):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # (a) Two Moons
    ax1_tm = axes[0]
    ax2_tm = ax1_tm.twinx()
    
    # GSQ Proposed (m2) Curves
    line1 = ax1_tm.plot(tau_data_tm["tau_values"], tau_data_tm["acc_test"], "o-", color=COLORS["ideal"],
                        lw=1.5, markersize=4.5, label="GSQ Accuracy (m2)")
    line2 = ax2_tm.plot(tau_data_tm["tau_values"], tau_data_tm["shock"], "s--", color=COLORS["gsq"],
                        lw=1.2, markersize=4.5, label="GSQ Shock (m2)")
    
    # Standard VQC (m0) Flat Baseline lines
    line3 = ax1_tm.plot(tau_data_tm["tau_values"], tau_data_tm["acc_m0"], "x:", color=COLORS["m0"],
                        lw=1.0, label="Standard Acc (m0)")
    line4 = ax2_tm.plot(tau_data_tm["tau_values"], tau_data_tm["shock_m0"], "x:", color="#d7191c",
                        lw=1.0, label="Standard Shock (m0)")
    
    ax1_tm.set_xlabel(r"Geometric hardening delay $\tau$")
    ax1_tm.set_ylabel("GSQ Test accuracy", color=COLORS["ideal"])
    ax2_tm.set_ylabel(r"Deployment shock $\Delta$", color=COLORS["gsq"])
    ax1_tm.tick_params(axis='y', labelcolor=COLORS["ideal"])
    ax2_tm.tick_params(axis='y', labelcolor=COLORS["gsq"])
    ax1_tm.set_title("(a) Two Moons: Hardening delay ablation")
    
    # Combined legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1_tm.legend(lines, labels, fontsize=6, loc="lower left")

    # (b) MNIST
    ax1_mn = axes[1]
    ax2_mn = ax1_mn.twinx()
    
    ax1_mn.plot(tau_data_mn["tau_values"], tau_data_mn["acc_test"], "o-", color=COLORS["ideal"],
                lw=1.5, markersize=4.5)
    ax2_mn.plot(tau_data_mn["tau_values"], tau_data_mn["shock"], "s--", color=COLORS["gsq"],
                lw=1.2, markersize=4.5)
    
    ax1_mn.plot(tau_data_mn["tau_values"], tau_data_mn["acc_m0"], "x:", color=COLORS["m0"], lw=1.0)
    ax2_mn.plot(tau_data_mn["tau_values"], tau_data_mn["shock_m0"], "x:", color="#d7191c", lw=1.0)
    
    ax1_mn.set_xlabel(r"Geometric hardening delay $\tau$")
    ax1_mn.set_ylabel("GSQ Test accuracy", color=COLORS["ideal"])
    ax2_mn.set_ylabel(r"Deployment shock $\Delta$", color=COLORS["gsq"])
    ax1_mn.tick_params(axis='y', labelcolor=COLORS["ideal"])
    ax2_mn.tick_params(axis='y', labelcolor=COLORS["gsq"])
    ax1_mn.set_title("(b) MNIST 0 vs 1: Hardening delay ablation")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        png_path = save_path.replace(".pdf", ".png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {save_path} and {png_path}")
    plt.close()


# ─────────────────────────────────────────────
# Main Loading & Execution
# ─────────────────────────────────────────────

def main(save_dir: str = "results", fig_dir: str = "figures"):
    os.makedirs(fig_dir, exist_ok=True)

    # --- 1. Load History & Configs ---
    hist_tm_path = os.path.join(save_dir, "two_moons", "train_history.json")
    hist_mn_path = os.path.join(save_dir, "mnist", "train_history.json")
    
    with open(hist_tm_path) as f:
        history_tm = json.load(f)
    with open(hist_mn_path) as f:
        history_mn = json.load(f)

    # Load tau configurations
    tau_tm = 0.6
    cfg_tm_path = os.path.join(save_dir, "two_moons", "config.json")
    if os.path.exists(cfg_tm_path):
        with open(cfg_tm_path) as f:
            tau_tm = json.load(f).get("tau", 0.6)

    tau_mn = 0.6
    cfg_mn_path = os.path.join(save_dir, "mnist", "config.json")
    if os.path.exists(cfg_mn_path):
        with open(cfg_mn_path) as f:
            tau_mn = json.load(f).get("tau", 0.6)

    # --- 2. Load Statistical Reports ---
    stat_tm_path = os.path.join(save_dir, "two_moons", "statistical_report.json")
    stat_mn_path = os.path.join(save_dir, "mnist", "statistical_report.json")
    
    with open(stat_tm_path) as f:
        stat_tm = json.load(f)
    with open(stat_mn_path) as f:
        stat_mn = json.load(f)

    print("\n>>> Redesigning and compiling all 3-way comparative publication figures...")

    # --- Figure 1: Training Curves ---
    plot_training_curves_multi(
        history_tm, history_mn, tau_tm=tau_tm, tau_mn=tau_mn,
        save_path=os.path.join(fig_dir, "fig1_training_curves.pdf")
    )

    # --- Figure 2: Grouped Distortion vs Shock (with error bars) ---
    plot_dfs_vs_delta_multi(
        stat_tm, stat_mn,
        save_path=os.path.join(fig_dir, "fig2_dfs_delta.pdf")
    )

    # --- Figure 3: Load & Plot K Ablation ---
    k_tm_path = os.path.join(save_dir, "two_moons", "k_ablation.json")
    k_mn_path = os.path.join(save_dir, "mnist", "k_ablation.json")
    with open(k_tm_path) as f:
        k_data_tm = json.load(f)
    with open(k_mn_path) as f:
        k_data_mn = json.load(f)
    
    plot_k_ablation_multi(
        k_data_tm, k_data_mn,
        save_path=os.path.join(fig_dir, "fig3_k_ablation.pdf")
    )

    # --- Figure 4: Load & Plot Tau Ablation ---
    tau_tm_path = os.path.join(save_dir, "two_moons", "tau_ablation.json")
    tau_mn_path = os.path.join(save_dir, "mnist", "tau_ablation.json")
    with open(tau_tm_path) as f:
        tau_data_tm = json.load(f)
    with open(tau_mn_path) as f:
        tau_data_mn = json.load(f)

    plot_tau_ablation_multi(
        tau_data_tm, tau_data_mn,
        save_path=os.path.join(fig_dir, "fig4_tau_ablation.pdf")
    )

    print(f"\nSUCCESS: Matplotlib compiled all 3-way comparative figures in '{fig_dir}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--fig_dir",  type=str, default="figures")
    args = parser.parse_args()
    main(args.save_dir, args.fig_dir)
