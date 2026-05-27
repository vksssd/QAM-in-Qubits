"""
run_experiments.py
------------------
Master orchestrator script for the GSQ multi-seed and multi-qubit study.

Trains and compares:
  - m0: Standard Variational Quantum Classifier (VQC) with direct grid rounding.
  - m1: Euclidean Trajectory K-Means quantization baseline.
  - m2: Proposed Geometric Spherical Quantization (GSQ) circular clustering + slerp.

Supports:
  - Multi-seed modes: `--mode smoke` (5 seeds) or `--mode full` (11 seeds).
  - Multi-qubit sweep: 4, 8, 12, and 16 qubits.
  - Adaptive epoch sizing: 40 epochs for smoke runs (blazing fast), 100 epochs for full runs.
  - Generates comprehensive qubit scaling reports.
"""

import os
import json
import subprocess
import sys
import argparse
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
    parser = argparse.ArgumentParser(description="GSQ Multi-Seed & Multi-Qubit Study Orchestrator")
    parser.add_argument("--mode", type=str, choices=["smoke", "full"], default="smoke",
                        help="Experimental mode: 'smoke' (5 seeds, 40 epochs) or 'full' (11 seeds, 100 epochs)")
    args = parser.parse_args()

    # Determine seeds and epochs based on mode
    if args.mode == "smoke":
        seeds = [42, 43, 44, 45, 46]
        epochs = 40
        print(f"\n>>> running SMOKE TEST SWEEP (5 seeds: {seeds}, {epochs} epochs) <<<")
    else:
        seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        epochs = 100
        print(f"\n>>> running FULL PUBLICATION SWEEP (11 seeds: {seeds}, {epochs} epochs) <<<")

    qubits_list = [4, 8, 12, 16]
    datasets = ["two_moons", "mnist"]

    # Master results dict for all qubit scaling comparisons
    master_comparison = {
        "qubit_counts": qubits_list,
        "mode": args.mode,
        "epochs": epochs,
        "two_moons": {"m0": {}, "euclidean": {}, "gsq": {}},
        "mnist": {"m0": {}, "euclidean": {}, "gsq": {}}
    }

    # Initialize master structure lists
    for ds in datasets:
        for method in ["m0", "euclidean", "gsq"]:
            master_comparison[ds][method] = {
                "accuracy": {"mean": [], "std": []},
                "D_FS": {"mean": [], "std": []},
                "delta": {"mean": [], "std": []}
            }

    print(f"\n{'='*80}")
    print("  GSQ MULTI-SEED & MULTI-QUBIT DEPTH STUDY  |  QISKIT AER NOISY EMULATION")
    print(f"  Qubits: {qubits_list}  |  Seeds: {len(seeds)}  |  Epochs: {epochs}")
    print(f"{'='*80}\n")

    for n_qubits in qubits_list:
        print(f"\n\n#################################################################")
        print(f"  SWEEPING QUBIT COUNT: N = {n_qubits} QUBITS")
        print(f"#################################################################\n")

        nsamples = 100 if n_qubits >= 12 else 300

        for ds in datasets:
            print(f"\n--- {ds.upper()} ({n_qubits} Qubits, {len(seeds)} seeds) ---")
            
            # Setup directories
            q_save_dir = os.path.join("results", ds, f"q{n_qubits}")
            os.makedirs(q_save_dir, exist_ok=True)

            rep_path = os.path.join(q_save_dir, "statistical_report.json")
            k_abl_path = os.path.join(q_save_dir, "k_ablation.json")
            tau_abl_path = os.path.join(q_save_dir, "tau_ablation.json")
            
            if os.path.exists(rep_path) and os.path.exists(k_abl_path) and os.path.exists(tau_abl_path):
                print(f"    [SKIP] Found completed results for N={n_qubits} qubits, {ds}. Loading saved data.")
                with open(rep_path, "r") as f:
                    rep_data = json.load(f)
                
                # Retrieve metrics and append directly to master_comparison
                metrics = rep_data["metrics"]
                for method, met_key in [("m0", "m0"), ("euclidean", "euclidean"), ("gsq", "gsq")]:
                    master_comparison[ds][method]["accuracy"]["mean"].append(metrics[f"{met_key}_accuracy"]["mean"])
                    master_comparison[ds][method]["accuracy"]["std"].append(metrics[f"{met_key}_accuracy"]["std"])
                    master_comparison[ds][method]["D_FS"]["mean"].append(metrics[f"{met_key}_dfs"]["mean"])
                    master_comparison[ds][method]["D_FS"]["std"].append(metrics[f"{met_key}_dfs"]["std"])
                    master_comparison[ds][method]["delta"]["mean"].append(metrics[f"{met_key}_shock"]["mean"])
                    master_comparison[ds][method]["delta"]["std"].append(metrics[f"{met_key}_shock"]["std"])
                continue

            acc_ideal, acc_m0, acc_euclid, acc_gsq = [], [], [], []
            dfs_m0, dfs_euclid, dfs_gsq = [], [], []
            shk_m0, shk_euclid, shk_gsq = [], [], []

            for seed in seeds:
                print(f"    Seed {seed} | Training & quantized deployment...")
                
                # 1. Train Standard VQC (m0)
                cfg_std = GSQConfig(
                    dataset=ds, n_qubits=n_qubits, epochs=epochs, n_samples=nsamples,
                    lambda_max=0.0, noise_model="none", seed=seed, save_dir=q_save_dir,
                    backend="lightning.qubit"
                )
                train(cfg_std)
                os.rename(
                    os.path.join(q_save_dir, "trained_weights.pt"),
                    os.path.join(q_save_dir, "standard_weights.pt")
                )

                # 2. Train GSQ VQC (m2)
                cfg_tr = GSQConfig(
                    dataset=ds, n_qubits=n_qubits, epochs=epochs, n_samples=nsamples,
                    noise_model="none", seed=seed, save_dir=q_save_dir,
                    backend="lightning.qubit"
                )
                train(cfg_tr)

                # 3. Quantize and Deploy under Qiskit Aer noisy QPU emulation (evaluates m0, m1, m2)
                cfg_de = GSQConfig(
                    dataset=ds, n_qubits=n_qubits, epochs=epochs, n_samples=nsamples,
                    noise_model="depolarizing", seed=seed, save_dir=q_save_dir
                )
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

            # Welch's t-test comparing GSQ (m2) vs. Standard VQC (m0)
            _, p_dfs = ttest_ind(dfs_m0, dfs_gsq, equal_var=False)
            _, p_shk = ttest_ind(shk_m0, shk_gsq, equal_var=False)
            _, p_acc = ttest_ind(acc_gsq, acc_m0, equal_var=False)

            # Cohen's d comparing GSQ (m2) vs. Standard VQC (m0)
            d_dfs = cohen_d(dfs_m0, dfs_gsq)
            d_shk = cohen_d(shk_m0, shk_gsq)
            d_acc = cohen_d(acc_gsq, acc_m0)

            # Save qubit-specific statistical report
            report = {
                "n_qubits": n_qubits,
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

            rep_path = os.path.join(q_save_dir, "statistical_report.json")
            with open(rep_path, "w") as f:
                json.dump(report, f, indent=2)

            print(f"    Saved N={n_qubits} report -> {rep_path}")
            print(f"    GSQ (m2) vs Standard (m0) significance:")
            print(f"        Acc: p={p_acc:.4f} (d={d_acc:.2f}) | Distortion: p={p_dfs:.4f} (d={d_dfs:.2f}) | Shock: p={p_shk:.4f} (d={d_shk:.2f})")

            # Append to master comparison dictionary for scale-up plotting
            for method, acc_arr, dfs_arr, shk_arr in [
                ("m0", acc_m0, dfs_m0, shk_m0),
                ("euclidean", acc_euclid, dfs_euclid, shk_euclid),
                ("gsq", acc_gsq, dfs_gsq, shk_gsq)
            ]:
                master_comparison[ds][method]["accuracy"]["mean"].append(float(np.mean(acc_arr)))
                master_comparison[ds][method]["accuracy"]["std"].append(float(np.std(acc_arr)))
                master_comparison[ds][method]["D_FS"]["mean"].append(float(np.mean(dfs_arr)))
                master_comparison[ds][method]["D_FS"]["std"].append(float(np.std(dfs_arr)))
                master_comparison[ds][method]["delta"]["mean"].append(float(np.mean(shk_arr)))
                master_comparison[ds][method]["delta"]["std"].append(float(np.std(shk_arr)))

            # ── Run Ablations for default seed (42) at this qubit count ──
            # K-ablation
            print(f"    Running K-ablation sweeps for N={n_qubits} qubits...")
            k_values = [2, 4, 6, 8, 10, 12, 16]
            shock_m0, shock_euclid, shock_gsq = [], [], []
            for K in k_values:
                cfg = GSQConfig(
                    dataset=ds, n_qubits=n_qubits, K=K, epochs=epochs, n_samples=nsamples,
                    noise_model="depolarizing", seed=42, save_dir=q_save_dir
                )
                res = deploy(cfg)
                shock_m0.append(res["m0"]["delta"])
                shock_euclid.append(res["euclidean"]["delta"])
                shock_gsq.append(res["gsq"]["delta"])
            
            k_abl_data = {"k_values": k_values, "shock_m0": shock_m0, "shock_euclid": shock_euclid, "shock_gsq": shock_gsq}
            k_abl_path = os.path.join(q_save_dir, "k_ablation.json")
            with open(k_abl_path, "w") as f:
                json.dump(k_abl_data, f, indent=2)

            # Tau-ablation
            print(f"    Running Tau-ablation sweeps for N={n_qubits} qubits...")
            tau_values = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9]
            acc_test, shock = [], []
            cfg_m0 = GSQConfig(
                dataset=ds, n_qubits=n_qubits, K=8, epochs=epochs, n_samples=nsamples,
                noise_model="depolarizing", seed=42, save_dir=q_save_dir
            )
            res_m0 = deploy(cfg_m0)
            acc_m0_val = res_m0["m0"]["accuracy"]
            shock_m0_val = res_m0["m0"]["delta"]

            for tau in tau_values:
                # Train GSQ
                cfg_tr = GSQConfig(
                    dataset=ds, n_qubits=n_qubits, tau=tau, epochs=epochs, n_samples=nsamples,
                    noise_model="none", seed=42, save_dir=q_save_dir,
                    backend="lightning.qubit"
                )
                train(cfg_tr)
                # Deploy
                cfg_de = GSQConfig(
                    dataset=ds, n_qubits=n_qubits, tau=tau, epochs=epochs, n_samples=nsamples,
                    noise_model="depolarizing", seed=42, save_dir=q_save_dir
                )
                res = deploy(cfg_de)
                acc_test.append(res["gsq"]["accuracy"])
                shock.append(res["gsq"]["delta"])

            tau_abl_data = {
                "tau_values": tau_values,
                "acc_m0": [acc_m0_val] * len(tau_values),
                "shock_m0": [shock_m0_val] * len(tau_values),
                "acc_test": acc_test,
                "shock": shock
            }
            tau_abl_path = os.path.join(q_save_dir, "tau_ablation.json")
            with open(tau_abl_path, "w") as f:
                json.dump(tau_abl_data, f, indent=2)

    # ── Save Master Multi-Qubit Comparison JSON ──
    master_comp_path = "results/multi_qubit_comparison.json"
    with open(master_comp_path, "w") as f:
        json.dump(master_comparison, f, indent=2)
    print(f"\n>>> Saved Master Multi-Qubit scaling results to {master_comp_path} <<<\n")

    # ── Step 6: Generate Scaling & Publication Figures ──
    print(">>> Executing plot_results.py to compile all 3-way figures and qubit scale-up curves...")
    try:
        subprocess.run([sys.executable, "plot_results.py"], check=True)
        print("\n=== SUCCESS: All multi-qubit publication figures successfully compiled in 'figures/' ===")
    except Exception as e:
        print(f"\nERROR: Failed to run plot_results.py: {e}")


if __name__ == "__main__":
    main()
