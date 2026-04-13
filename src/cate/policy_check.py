from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TAU_LE_THR = "tau_le_thr"
TAU_GE_THR = "tau_ge_thr"

def policy_decision(tau: pd.Series, threshold: float, rule: str = "tau_le_thr") -> np.ndarray:
    """
    rule:
      - 'tau_le_thr': treat if tau <= threshold  (coherent with growing treat_rate with the threshold)
      - 'tau_ge_thr': treat if tau >= threshold
    """
    if rule == "tau_le_thr":
        return (tau.values <= threshold).astype(int)
    elif rule == "tau_ge_thr":
        return (tau.values >= threshold).astype(int)
    elif rule == "top_frac_benefit":
        # threshold is interpreted as top fraction, e.g. 0.2 = top 20% most benefit patients
        n = len(tau)
        n_treat = int(np.ceil(n * threshold))
        top_idx = np.argsort(tau.values)[:n_treat]  # indices of top n_treat patients
        a_pi = np.zeros(n, dtype=int)
        a_pi[top_idx] = 1
        return a_pi
    else:
        raise ValueError("rule must be 'tau_le_thr', 'tau_ge_thr' or 'top_frac_benefit'")

def make_risk_tau_policy_plots(
    df: pd.DataFrame,
    threshold: float,
    n_bins: int = 5,
    rule: str = "tau_le_thr",
    tau_col: str = "tau_hat",
    mu0_col: str = "mu0_hat",
    mu1_col: str = "mu1_hat",
    out_dir: Path = None,
    outcome_nice_name: str = "Mortalità",
):
    """
    Produces:
      1) boxplot of tau_hat over risk quantiles mu0_hat
      2) line plot: treat rate for each bin
      3) bar plot: estimated mortality under policy vs treat-none (plugin)
      4) bar plot: estimated mortality under policy vs treat-all (plugin)
      5) bar plot: expected benefit contribution per bin
    """
    out_dir = out_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in [tau_col, mu0_col, mu1_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    d = df[[tau_col, mu0_col, mu1_col]].dropna().copy()

    # bins / base risk (mu0_hat)
    d["risk_bin"] = pd.qcut(d[mu0_col], q=n_bins, labels=False, duplicates="drop")
    n_bins_eff = int(d["risk_bin"].max()) + 1

    # policy decision and risk under policy (plugin)
    a_pi = policy_decision(d[tau_col], threshold=threshold, rule=rule)
    d["a_pi"] = a_pi
    d["mu_pi"] = np.where(d["a_pi"].values == 1, d[mu1_col].values, d[mu0_col].values)

    # aggregations per bin
    g = d.groupby("risk_bin", observed=True)

    summary = g.agg(
        n=(mu0_col, "size"),
        risk0_mean=(mu0_col, "mean"),
        tau_mean=(tau_col, "mean"),
        tau_median=(tau_col, "median"),
        treat_rate=("a_pi", "mean"),
        mort_policy=("mu_pi", "mean"),
        mort_none=(mu0_col, "mean"),
        mort_treat_all=(mu1_col, "mean"),
    ).reset_index()
    # Policy delta vs treat-all
    summary["delta_policy_vs_treat_all"] = summary["mort_policy"] - summary["mort_treat_all"]
    # Policy delta vs treat-none
    summary["delta_policy_vs_none"] = summary["mort_policy"] - summary["mort_none"]  # <0 = better
    # Absolute contribute (how much it lowers that bin on average)
    summary["benefit_abs"] = -summary["delta_policy_vs_none"]  # positive = benefit
    # total contribution to improvement (weighted by bin size)
    summary["benefit_total"] = summary["benefit_abs"] * summary["n"]

    # --------- Plot 1: tau distribution by risk bin (boxplot)
    tau_by_bin = [d.loc[d["risk_bin"] == b, tau_col].values for b in range(n_bins_eff)]

    plt.figure()
    plt.boxplot(tau_by_bin, showfliers=False)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(f"Rischio baseline bins (quantili di {mu0_col})")
    plt.ylabel(tau_col)
    plt.title("Distribuzione di τ̂ per quantili di rischio baseline")
    plt.savefig(out_dir / "policy_tau_by_risk_bin.png")
    plt.close()

    # --------- Plot 2: treat rate per bin
    plt.figure()
    plt.plot(np.arange(n_bins_eff) + 1, summary["treat_rate"].values, marker="o")
    plt.ylim(0, 1)
    plt.xlabel(f"Bin rischio baseline (1=più basso → {n_bins_eff}=più alto)")
    plt.ylabel("Treat rate della policy")
    plt.title("Quanto tratta la policy nei diversi livelli di rischio")
    plt.savefig(out_dir / "policy_treat_rate_by_risk_bin.png")
    plt.close()

    # --------- Plot 3: policy mortality vs none (plugin)
    plt.figure()
    x = np.arange(n_bins_eff)
    plt.bar(x - 0.2, summary["mort_none"].values, width=0.4, label="Treat-none (mu0)")
    plt.bar(x + 0.2, summary["mort_policy"].values, width=0.4, label="Policy (mu_pi)")
    plt.xlabel(f"Bin rischio baseline (1=più basso → {n_bins_eff}=più alto)")
    plt.ylabel(f"{outcome_nice_name} stimata (plugin)")
    plt.title(f"{outcome_nice_name} stimata per bin: policy vs treat-none")
    plt.legend()
    plt.grid()
    plt.savefig(out_dir / "policy_outcome_by_risk_bin.png")
    plt.close()

    # --------- Plot 4: policy mortality vs treat-all (plugin)
    plt.figure()
    x = np.arange(n_bins_eff)
    plt.bar(x - 0.2, summary["mort_treat_all"].values, width=0.4, label="Treat-all (mu1)")
    plt.bar(x + 0.2, summary["mort_policy"].values, width=0.4, label="Policy (mu_pi)")
    plt.xlabel(f"Bin rischio baseline (1=più basso → {n_bins_eff}=più alto)")
    plt.ylabel(f"{outcome_nice_name} stimata (plugin)")
    plt.title(f"{outcome_nice_name} stimata per bin: policy vs treat-all")
    plt.legend()
    plt.grid()
    plt.savefig(out_dir / "policy_outcome_vs_treat_all_by_risk_bin.png")
    plt.close()

    # --------- Plot 5: contribution (benefit_total)
    plt.figure()
    plt.bar(np.arange(n_bins_eff), summary["benefit_total"].values)
    plt.xlabel(f"Bin rischio baseline (1=più basso → {n_bins_eff}=più alto)")
    plt.ylabel("Contributo totale al miglioramento (≈ n * Δ)")
    plt.title("Da quali bins arriva il miglioramento della policy")
    plt.savefig(out_dir / "policy_benefit_contribution_by_risk_bin.png")
    plt.close()

    return summary

# USE EXAMPLE:
# df = pd.read_parquet(".../predictions.parquet")
# summary = make_risk_tau_policy_plots(df, threshold=0.0216, n_bins=5, rule="tau_le_thr")
# print(summary)