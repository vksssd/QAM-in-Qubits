"""
run_experiments.py
------------------
Master orchestrator script for the GSQ project.

Trains and compares three models:
  - m0: Standard VQC (pure task loss training, grid rounding quantization)
  - m1: Trajectory Euclidean K-means (our GSQ training but Euclidean distance clustering)
  - m2: GSQ Circular (our GSQ training, circular clustering, soft toroidal slerp projection)

Runs the following pipeline:
  1. Noisy Default Run: Trains Standard VQC (m0) and GSQ VQC (m2), and deploys m0, m1, m2 under QPU noise.
  2. Multi-Seed Sweeps: Executes 5 seeds [42, 43, 44, 45, 46], computing Welch's t-test and Cohen's d comparing m2 vs m0.
  3. Noisy K-Ablation Sweep: Sweeps K in [2, 4, 6, 8, 10, 12, 16] under depolarizing noise for m0, m1, m2.
  4. Noisy Tau-Ablation Sweep: Sweeps tau in [0.1, 0.3, 0.5, 0.6, 0.7, 0.9] under depolarizing noise (with m0 as baseline).
  5. Restore Default Checkpoints: Restores default noisy models on disk.
  6. Runs plot_results.py to compile comparative figures with standard deviation bands/ribbons.
"""

import os
import json
import subprocess
import sys
import numpy as np
from scipy.stats import ttest_ind

from config import GSQConfig
from train import train
from deploy import deploy


def cohen_d(x1, x2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(x1), len(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(x1) - np.mean(x2)) / pooled_sd)


def main():
    print(f"\n{'='*80}")
    print("  GSQ 3-WAY COMPARISON & STATISTICAL sweeps  |  TWO MOONS & MNIST")
    print("  Comparing: m0 (Standard VQC) vs. m1 (Euclidean VQC) vs. m2 (GSQ Proposed)")
    print(f"{'='*80}\n")

    os.makedirs("results/two_moons", exist_ok=True)
    os.makedirs("results/mnist", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    datasets = ["two_moons", "mnist"]
    seeds = [42, 43, 44, 45, 46]

    # ── Step 1: Default Noisy Run ──
    print(">>> [1/6] Running default noisy QPU training & deployment...")
    for ds in datasets:
        print(f"\n--- Default Noisy Run: {ds} (K=8, alpha=0.5) ---")
        
        # 1. Train standard VQC (m0)
        print("Training Standard VQC (m0)...")
        cfg_std = GSQConfig(dataset=ds, epochs=100, n_samples=400, lambda_max=0.0, noise_model="none", seed=42)
        train(cfg_std)
        os.rename(
            os.path.join(cfg_std.save_dir, "trained_weights.pt"),
            os.path.join(cfg_std.save_dir, "standard_weights.pt")
        )
        
        # 2. Train GSQ VQC (m2)
        print("Training GSQ VQC (m2)...")
        cfg_train = GSQConfig(dataset=ds, epochs=100, n_samples=400, noise_model="none", seed=42)
        train(cfg_train)
        
        # 3. Deploy with depolarizing gate noise & phase relaxation
        cfg_deploy = GSQConfig(dataset=ds, epochs=100, n_samples=400, noise_model="depolarizing", seed=42)
        deploy(cfg_deploy)

    # ── Step 2: Multi-Seed Statistical Significance Sweep ──
    print("\n>>> [2/6] Running 5-seed statistical sweep under depolarizing noise...")
    for ds in datasets:
        print(f"\n--- Statistical Sweep for {ds} (Seeds: {seeds}) ---")
        acc_ideal, acc_m0, acc_euclid, acc_gsq = [], [], [], []
        dfs_m0, dfs_euclid, dfs_gsq = [], [], []
        shk_m0, shk_euclid, shk_gsq = [], [], []

        for seed in seeds:
            print(f"    Running Seed {seed}...")
            # Train standard (m0)
            cfg_std = GSQConfig(dataset=ds, epochs=100, n_samples=400, lambda_max=0.0, noise_model="none", seed=seed)
            train(cfg_std)
            os.rename(
                os.path.join(cfg_std.save_dir, "trained_weights.pt"),
                os.path.join(cfg_std.save_dir, "standard_weights.pt")
            )
            
            # Train GSQ (m2)
            cfg_tr = GSQConfig(dataset=ds, epochs=100, n_samples=400, noise_model="none", seed=seed)
            train(cfg_tr)
            
            # Deploy noisy (evaluates m0, m1, m2)
            cfg_de = GSQConfig(dataset=ds, epochs=100, n_samples=400, noise_model="depolarizing", seed=seed)
            res = deploy(cfg_de)
            
            acc_ideal.append(res["ideal"]["accuracy"])
            acc_m0.append(res["m0"]["accuracy"])
            acc_euclid.append(res["euclidean"]["accuracy"])
            acc_gsq.append(res["gsq"]["accuracy"])
            
            dfs_m0.append(res["m0"]["D_FS"])
            dfs_euclid.append(res["euclidean"]["D_FS"])
            dfs_gsq.append(res["gsq"]["D_FS"])
            
            shk_m0.append(res["m0"]["delta"])
            shk_euclid.append(res["euclidean"]["delta"])
            shk_gsq.append(res["gsq"]["delta"])

        # Welch's t-test comparing proposed GSQ (m2) vs. Standard VQC (m0)
        _, p_dfs = ttest_ind(dfs_m0, dfs_gsq, equal_var=False)
        _, p_shk = ttest_ind(shk_m0, shk_gsq, equal_var=False)
        _, p_acc = ttest_ind(acc_gsq, acc_m0, equal_var=False)

        # Cohen's d comparing proposed GSQ (m2) vs. Standard VQC (m0)
        d_dfs = cohen_d(dfs_m0, dfs_gsq)
        d_shk = cohen_d(shk_m0, shk_gsq)
        d_acc = cohen_d(acc_gsq, acc_m0)

        report = {
            "dataset": ds,
            "seeds": seeds,
            "metrics": {
                "ideal_accuracy": {"mean": float(np.mean(acc_ideal)), "std": float(np.std(acc_ideal))},
                "m0_accuracy": {"mean": float(np.mean(acc_m0)), "std": float(np.std(acc_m0))},
                "euclidean_accuracy": {"mean": float(np.mean(acc_euclid)), "std": float(np.std(acc_euclid))},
                "gsq_accuracy": {"mean": float(np.mean(acc_gsq)), "std": float(np.std(acc_gsq))},
                "m0_dfs": {"mean": float(np.mean(dfs_m0)), "std": float(np.std(dfs_m0))},
                "euclidean_dfs": {"mean": float(np.mean(dfs_euclid)), "std": float(np.std(dfs_euclid))},
                "gsq_dfs": {"mean": float(np.mean(dfs_gsq)), "std": float(np.std(dfs_gsq))},
                "m0_shock": {"mean": float(np.mean(shk_m0)), "std": float(np.std(shk_m0))},
                "euclidean_shock": {"mean": float(np.mean(shk_euclid)), "std": float(np.std(shk_euclid))},
                "gsq_shock": {"mean": float(np.mean(shk_gsq)), "std": float(np.std(shk_gsq))},
            },
            "statistical_tests": {
                "dfs": {"p_value": float(p_dfs) if not np.isnan(p_dfs) else 1.0, "cohen_d": float(d_dfs)},
                "shock": {"p_value": float(p_shk) if not np.isnan(p_shk) else 1.0, "cohen_d": float(d_shk)},
                "accuracy": {"p_value": float(p_acc) if not np.isnan(p_acc) else 1.0, "cohen_d": float(d_acc)}
            }
        }

        rep_path = f"results/{ds}/statistical_report.json"
        with open(rep_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"    Saved statistical report -> {rep_path}")
        print(f"    GSQ (m2) vs Standard VQC (m0) significance tests:")
        print(f"        Fubini-Study Distortion: p={p_dfs:.4f} (Cohen's d={d_dfs:.2f})")
        print(f"        Deployment Shock:        p={p_shk:.4f} (Cohen's d={d_shk:.2f})")

    # ── Step 3: Noisy K-Ablation Sweep ──
    print("\n>>> [3/6] Running K-ablation sweeps under depolarizing noise for m0, m1, m2...")
    k_values = [2, 4, 6, 8, 10, 12, 16]
    for ds in datasets:
        print(f"\n--- Sweeping K for {ds} under QPU Noise ---")
        shock_m0 = []
        shock_gsq = []
        shock_euclid = []
        for K in k_values:
            cfg = GSQConfig(dataset=ds, K=K, n_samples=400, noise_model="depolarizing", seed=42)
            res = deploy(cfg)
            shock_m0.append(res["m0"]["delta"])
            shock_euclid.append(res["euclidean"]["delta"])
            shock_gsq.append(res["gsq"]["delta"])
            print(f"    K={K:>2d} | m0 Shock: {res['m0']['delta']:.4f} | Euclid (m1) Shock: {res['euclidean']['delta']:.4f} | GSQ (m2) Shock: {res['gsq']['delta']:.4f}")

        k_abl_data = {
            "k_values": k_values,
            "shock_m0": shock_m0,
            "shock_euclid": shock_euclid,
            "shock_gsq": shock_gsq
        }
        abl_path = f"results/{ds}/k_ablation.json"
        with open(abl_path, "w") as f:
            json.dump(k_abl_data, f, indent=2)
        print(f"    Saved sweep -> {abl_path}")

    # ── Step 4: Noisy Tau-Ablation Sweep ──
    print("\n>>> [4/6] Running Tau-ablation sweeps under depolarizing noise...")
    tau_values = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9]
    for ds in datasets:
        print(f"\n--- Sweeping tau for {ds} under QPU Noise ---")
        acc_test = []
        shock = []
        # m0 remains flat constant since standard VQC is trained with lambda=0 (independent of tau)
        cfg_m0 = GSQConfig(dataset=ds, K=8, n_samples=400, noise_model="depolarizing", seed=42)
        res_m0 = deploy(cfg_m0)
        acc_m0_val = res_m0["m0"]["accuracy"]
        shock_m0_val = res_m0["m0"]["delta"]
        
        for tau in tau_values:
            # Train GSQ
            cfg_tr = GSQConfig(dataset=ds, tau=tau, epochs=60, n_samples=400, noise_model="none", seed=42)
            train(cfg_tr)
            
            # Deploy noisy
            cfg_de = GSQConfig(dataset=ds, tau=tau, epochs=60, n_samples=400, noise_model="depolarizing", seed=42)
            res = deploy(cfg_de)
            acc_test.append(res["gsq"]["accuracy"])
            shock.append(res["gsq"]["delta"])
            print(f"    tau={tau:.1f} | Test Acc: {res['gsq']['accuracy']:.3f} | Shock: {res['gsq']['delta']:.4f}")

        tau_abl_data = {
            "tau_values": tau_values,
            "acc_m0": [acc_m0_val] * len(tau_values),
            "shock_m0": [shock_m0_val] * len(tau_values),
            "acc_test": acc_test,
            "shock": shock
        }
        abl_path = f"results/{ds}/tau_ablation.json"
        with open(abl_path, "w") as f:
            json.dump(tau_abl_data, f, indent=2)
        print(f"    Saved sweep -> {abl_path}")

    # ── Step 5: Restore Default Noisy Checkpoints ──
    print("\n>>> [5/6] Restoring final default noisy checkpoints (tau=0.6, epochs=100)...")
    for ds in datasets:
        # Standard VQC (m0)
        cfg_std = GSQConfig(dataset=ds, epochs=100, n_samples=400, lambda_max=0.0, noise_model="none", seed=42)
        train(cfg_std)
        os.rename(
            os.path.join(cfg_std.save_dir, "trained_weights.pt"),
            os.path.join(cfg_std.save_dir, "standard_weights.pt")
        )
        # GSQ VQC (m2)
        cfg_tr = GSQConfig(dataset=ds, epochs=100, n_samples=400, noise_model="none", tau=0.6, seed=42)
        train(cfg_tr)
        cfg_de = GSQConfig(dataset=ds, epochs=100, n_samples=400, noise_model="depolarizing", tau=0.6, seed=42)
        deploy(cfg_de)

    # ── Step 6: Generate Publication Figures ──
    print("\n>>> [6/6] Executing plot_results.py to compile 3-way figures...")
    try:
        subprocess.run([sys.executable, "plot_results.py"], check=True)
        print("\n=== SUCCESS: All 3-way comparative figures successfully compiled in 'figures/' ===")
    except Exception as e:
        print(f"\nERROR: Failed to run plot_results.py: {e}")


if __name__ == "__main__":
    main()
