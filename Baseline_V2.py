"""
QMC Toy Model V2 — Phase 1: Single Node Proof of Concept
=========================================================

A single node carries a distribution ψ(t) over M discrete bins.
Dynamics: discretised Fokker-Planck with diffusion (𝒯) and repulsive potential (𝒩).
Comparison: distributed (ψ over M bins) vs punctual (δ-distribution).

Tests:
  1. Necessity of distribution: does ψ distributed maintain viability where punctual collapses?
  2. Variance corridor: does 0 < Var(ψ) < Var_crit emerge for a range of D?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional, Dict, Any


# ============================================================
# Configuration
# ============================================================

@dataclass
class V2Phase1Config:
    # Time
    T: int = 500
    dt: float = 0.05

    # Configurational space
    M: int = 20                    # number of bins in Θ
    theta_min: float = 0.0         # lower bound of Θ
    theta_max: float = 1.0         # upper bound of Θ

    # Dynamics
    D: float = 0.02                # diffusion coefficient (instantiates 𝒯)
    V_strength: float = 1.0        # strength of repulsive potential V_𝒩
    V_var_min: float = 0.005       # Var target below which V_𝒩 repels (KNV 5)
    V_var_max: float = 0.08        # Var target above which V_𝒩 repels (KNV 6)

    # Initial distribution
    init_mode: str = "narrow"      # "narrow" (near-punctual), "gaussian", "uniform"
    init_center: float = 0.5       # center of initial distribution
    init_width: float = 0.02       # width (std) of initial Gaussian

    # Viability thresholds (for analysis)
    var_floor: float = 0.001       # below this, consider collapsed
    H_floor: float = 0.05          # below this, consider entropically collapsed

    # Lethargy window
    lethargy_window: int = 30

    # Random seed
    seed: int = 42


# ============================================================
# Configurational space
# ============================================================

def make_theta_grid(cfg: V2Phase1Config) -> np.ndarray:
    """Uniform grid over Θ = [theta_min, theta_max], M bins."""
    return np.linspace(cfg.theta_min, cfg.theta_max, cfg.M)


def bin_width(cfg: V2Phase1Config) -> float:
    return (cfg.theta_max - cfg.theta_min) / (cfg.M - 1)


# ============================================================
# Initial distributions
# ============================================================

def init_distribution(cfg: V2Phase1Config, theta: np.ndarray) -> np.ndarray:
    """Create initial distribution ψ(0) over theta grid."""
    if cfg.init_mode == "narrow":
        # Near-punctual: very narrow Gaussian (approximates δ)
        psi = np.exp(-0.5 * ((theta - cfg.init_center) / (0.5 * bin_width(cfg)))**2)
    elif cfg.init_mode == "gaussian":
        psi = np.exp(-0.5 * ((theta - cfg.init_center) / cfg.init_width)**2)
    elif cfg.init_mode == "uniform":
        psi = np.ones_like(theta)
    else:
        raise ValueError(f"Unknown init_mode: {cfg.init_mode}")

    # Normalise
    psi = psi / (np.sum(psi) * bin_width(cfg))
    return psi


def init_punctual(cfg: V2Phase1Config, theta: np.ndarray) -> np.ndarray:
    """Create punctual (δ) distribution: all mass on nearest bin to init_center."""
    psi = np.zeros_like(theta)
    idx = np.argmin(np.abs(theta - cfg.init_center))
    psi[idx] = 1.0 / bin_width(cfg)
    return psi


# ============================================================
# Observables (Chapter 2)
# ============================================================

def compute_expectation(psi: np.ndarray, theta: np.ndarray, dtheta: float) -> float:
    return float(np.sum(theta * psi * dtheta))


def compute_variance(psi: np.ndarray, theta: np.ndarray, dtheta: float) -> float:
    mu = compute_expectation(psi, theta, dtheta)
    return float(np.sum((theta - mu)**2 * psi * dtheta))


def compute_entropy(psi: np.ndarray, dtheta: float) -> float:
    """Morphodynamic entropy H(ψ) = -∫ ψ ln ψ dθ."""
    psi_safe = np.clip(psi * dtheta, 1e-30, None)  # probability per bin
    return float(-np.sum(psi_safe * np.log(psi_safe)))


def compute_entropy_max(var: float) -> float:
    """Maximum entropy for given variance (Gaussian): ½ ln(2πeσ²)."""
    if var <= 0:
        return 0.0
    return 0.5 * np.log(2.0 * np.pi * np.e * var)


def compute_eta_H(H: float, var: float) -> float:
    """Entropic efficiency η_H = H / H_max(Var)."""
    H_max = compute_entropy_max(var)
    if H_max <= 0:
        return 0.0
    return min(float(H / H_max), 1.0)


def compute_lethargy(var_now: float, var_past: float) -> float:
    """ℒ = clamp₀¹(1 - Var(t)/Var(t-w))."""
    if var_past <= 0:
        return 1.0
    ratio = var_now / var_past
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


# ============================================================
# Repulsive potential V_𝒩 (heuristic)
# ============================================================

def compute_V_potential(theta: np.ndarray, psi: np.ndarray, dtheta: float,
                         cfg: V2Phase1Config) -> np.ndarray:
    """
    Heuristic V_𝒩: repels distribution when Var approaches 0 or Var_crit.
    Returns a force field F_𝒩(θ) = -dV/dθ acting on each bin.

    The potential creates:
    - Repulsion from collapse (Var → 0): pushes outward from center
    - Repulsion from dispersion (Var → Var_crit): pushes inward toward center
    """
    var = compute_variance(psi, theta, dtheta)
    mu = compute_expectation(psi, theta, dtheta)

    # Distance of each bin from center of distribution
    displacement = theta - mu

    # Collapse proximity: how close Var is to 0
    collapse_proximity = np.exp(-var / cfg.V_var_min) if var > 0 else 1.0

    # Dispersion proximity: how close Var is to Var_crit
    if var < cfg.V_var_max:
        dispersion_proximity = 0.0
    else:
        dispersion_proximity = np.exp(-(cfg.V_var_max - var)**2 / (0.5 * cfg.V_var_max**2))

    # Force: outward when collapse threatens, inward when dispersion threatens
    F = cfg.V_strength * (
        collapse_proximity * displacement            # push outward (widen)
        - dispersion_proximity * displacement         # push inward (narrow)
    )

    return F


# ============================================================
# Fokker-Planck dynamics (discretised)
# ============================================================

def fokker_planck_step(psi: np.ndarray, theta: np.ndarray, dtheta: float,
                       cfg: V2Phase1Config, rng: np.random.Generator) -> np.ndarray:
    """
    One step of discretised Fokker-Planck:
      ψ(t+1) = ψ(t) + D·Δ_θ ψ - ∂_θ[F_𝒩·ψ] + noise

    Uses upwind scheme for the drift term to preserve positivity.
    """
    M = len(psi)

    # --- Diffusion term: D · Δ_θ ψ (central differences) ---
    laplacian = np.zeros(M)
    laplacian[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dtheta**2
    # Reflecting boundaries
    laplacian[0] = (psi[1] - psi[0]) / dtheta**2
    laplacian[-1] = (psi[-2] - psi[-1]) / dtheta**2

    diffusion = cfg.D * laplacian

    # --- Drift term: -∂_θ[F_𝒩 · ψ] (upwind scheme) ---
    F = compute_V_potential(theta, psi, dtheta, cfg)
    flux = F * psi

    drift = np.zeros(M)
    for i in range(1, M - 1):
        # Upwind: use backward difference if F > 0, forward if F < 0
        if F[i] >= 0:
            drift[i] = -(flux[i] - flux[i-1]) / dtheta
        else:
            drift[i] = -(flux[i+1] - flux[i]) / dtheta
    # Boundaries: no flux
    drift[0] = 0.0
    drift[-1] = 0.0

    # --- Small structural noise (granular stratum contribution) ---
    noise = rng.normal(0.0, cfg.D * 0.1, M) * psi

    # --- Update ---
    psi_new = psi + cfg.dt * (diffusion + drift + noise)

    # --- Enforce positivity ---
    psi_new = np.maximum(psi_new, 0.0)

    # --- Renormalise (conservation of presence) ---
    total = np.sum(psi_new) * dtheta
    if total > 0:
        psi_new = psi_new / total
    else:
        # Fallback: reset to uniform
        psi_new = np.ones(M) / ((cfg.theta_max - cfg.theta_min))

    return psi_new


# ============================================================
# Punctual dynamics (V1-like on single node)
# ============================================================

def punctual_step(tau: float, cfg: V2Phase1Config, rng: np.random.Generator) -> float:
    """
    Single-node scalar dynamics (V1 analogue):
      τ(t+1) = τ(t) + reflexive + noise
    No coupling (single node). Same noise amplitude as V1.
    """
    reflexive = -0.15 * (tau - 0.5)
    noise = rng.normal(0.0, 0.028)
    tau_new = tau + cfg.dt * (reflexive + noise)
    return float(np.clip(tau_new, 0.05, 0.95))


# ============================================================
# Main simulation
# ============================================================

def run_distributed(cfg: V2Phase1Config) -> Dict[str, Any]:
    """Run distributed simulation: ψ(t) over M bins."""
    rng = np.random.default_rng(cfg.seed)
    theta = make_theta_grid(cfg)
    dtheta = bin_width(cfg)

    psi = init_distribution(cfg, theta)

    # History arrays
    var_history = np.zeros(cfg.T)
    H_history = np.zeros(cfg.T)
    eta_H_history = np.zeros(cfg.T)
    lethargy_history = np.zeros(cfg.T)
    expectation_history = np.zeros(cfg.T)
    psi_snapshots = []

    for t in range(cfg.T):
        # Observables
        var = compute_variance(psi, theta, dtheta)
        H = compute_entropy(psi, dtheta)
        eta_H = compute_eta_H(H, var)
        mu = compute_expectation(psi, theta, dtheta)

        # Lethargy
        if t >= cfg.lethargy_window:
            leth = compute_lethargy(var, var_history[t - cfg.lethargy_window])
        else:
            leth = 0.0

        var_history[t] = var
        H_history[t] = H
        eta_H_history[t] = eta_H
        lethargy_history[t] = leth
        expectation_history[t] = mu

        # Snapshots at key times
        if t in [0, cfg.T // 4, cfg.T // 2, 3 * cfg.T // 4, cfg.T - 1]:
            psi_snapshots.append((t, psi.copy()))

        # Step
        psi = fokker_planck_step(psi, theta, dtheta, cfg, rng)

    return {
        "config": cfg,
        "theta": theta,
        "var_history": var_history,
        "H_history": H_history,
        "eta_H_history": eta_H_history,
        "lethargy_history": lethargy_history,
        "expectation_history": expectation_history,
        "psi_snapshots": psi_snapshots,
        "final_var": float(var_history[-1]),
        "final_H": float(H_history[-1]),
        "final_eta_H": float(eta_H_history[-1]),
        "viable_fraction": float(np.mean(
            (var_history > cfg.var_floor) & (H_history > cfg.H_floor)
        )),
    }


def run_punctual(cfg: V2Phase1Config) -> Dict[str, Any]:
    """Run punctual simulation: scalar τ (V1 analogue, single node)."""
    rng = np.random.default_rng(cfg.seed)
    tau = cfg.init_center

    tau_history = np.zeros(cfg.T)
    var_history = np.zeros(cfg.T)  # Var = 0 by construction
    H_history = np.zeros(cfg.T)    # H = 0 by construction

    for t in range(cfg.T):
        tau_history[t] = tau
        var_history[t] = 0.0
        H_history[t] = 0.0
        tau = punctual_step(tau, cfg, rng)

    return {
        "config": cfg,
        "tau_history": tau_history,
        "var_history": var_history,
        "H_history": H_history,
        "final_tau": float(tau_history[-1]),
        "viable_fraction": 0.0,  # Punctual: H = 0 always → never entropically viable
    }


# ============================================================
# Plotting
# ============================================================

def plot_phase1_comparison(dist_result: Dict, punct_result: Dict,
                            title: str = "QMC V2 Phase 1 — Distributed vs Punctual",
                            save_path: Optional[str] = None) -> None:
    """6-panel comparison: distributed observables + punctual baseline."""
    cfg = dist_result["config"]
    T = cfg.T
    timesteps = np.arange(T)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(title, fontsize=16, y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)

    ax_var = fig.add_subplot(gs[0, 0])
    ax_H = fig.add_subplot(gs[0, 1])
    ax_eta = fig.add_subplot(gs[1, 0])
    ax_leth = fig.add_subplot(gs[1, 1])
    ax_dist = fig.add_subplot(gs[2, 0])
    ax_expect = fig.add_subplot(gs[2, 1])

    # --- Variance ---
    ax_var.plot(timesteps, dist_result["var_history"], lw=2, label="Distributed", color="blue")
    ax_var.axhline(cfg.V_var_min, ls="--", color="red", lw=1.2, label=f"Var_min={cfg.V_var_min}")
    ax_var.axhline(cfg.V_var_max, ls="--", color="orange", lw=1.2, label=f"Var_max={cfg.V_var_max}")
    ax_var.axhline(cfg.var_floor, ls=":", color="gray", lw=1, label=f"Var_floor={cfg.var_floor}")
    ax_var.set_title("Variance Var(ψ)")
    ax_var.set_xlabel("Time steps")
    ax_var.set_ylabel("Var")
    ax_var.legend(fontsize=8)
    ax_var.set_yscale("log")

    # --- Entropy ---
    ax_H.plot(timesteps, dist_result["H_history"], lw=2, label="Distributed", color="blue")
    ax_H.axhline(cfg.H_floor, ls=":", color="gray", lw=1, label=f"H_floor={cfg.H_floor}")
    ax_H.set_title("Morphodynamic Entropy H(ψ)")
    ax_H.set_xlabel("Time steps")
    ax_H.set_ylabel("H")
    ax_H.legend(fontsize=8)

    # --- Entropic efficiency ---
    ax_eta.plot(timesteps, dist_result["eta_H_history"], lw=2, color="green")
    ax_eta.set_title("Entropic Efficiency η_H")
    ax_eta.set_xlabel("Time steps")
    ax_eta.set_ylabel("η_H")
    ax_eta.set_ylim(-0.05, 1.05)

    # --- Lethargy ---
    ax_leth.plot(timesteps, dist_result["lethargy_history"], lw=2, color="purple")
    ax_leth.set_title("Lethargy ℒ(t)")
    ax_leth.set_xlabel("Time steps")
    ax_leth.set_ylabel("ℒ")
    ax_leth.set_ylim(-0.05, 1.05)

    # --- Distribution snapshots ---
    theta = dist_result["theta"]
    for t_snap, psi_snap in dist_result["psi_snapshots"]:
        ax_dist.plot(theta, psi_snap, label=f"t={t_snap}", lw=1.5)
    ax_dist.set_title("Distribution ψ(θ) at selected times")
    ax_dist.set_xlabel("θ")
    ax_dist.set_ylabel("ψ(θ)")
    ax_dist.legend(fontsize=8)

    # --- Expectation vs punctual ---
    ax_expect.plot(timesteps, dist_result["expectation_history"], lw=2,
                   label="Distributed 𝔼[ψ]", color="blue")
    ax_expect.plot(timesteps, punct_result["tau_history"], lw=2,
                   label="Punctual τ", color="red", ls="--")
    ax_expect.set_title("Expectation (Distributed) vs τ (Punctual)")
    ax_expect.set_xlabel("Time steps")
    ax_expect.set_ylabel("Value")
    ax_expect.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_variance_sweep(sweep_results: list,
                         title: str = "QMC V2 Phase 1 — Variance Corridor (D sweep)",
                         save_path: Optional[str] = None) -> None:
    """Plot final Var and H as function of D."""
    D_values = [r["D"] for r in sweep_results]
    final_vars = [r["final_var"] for r in sweep_results]
    final_Hs = [r["final_H"] for r in sweep_results]
    viable_fracs = [r["viable_fraction"] for r in sweep_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, y=1.02)

    axes[0].plot(D_values, final_vars, "o-", lw=2, color="blue")
    axes[0].set_xlabel("Diffusion coefficient D")
    axes[0].set_ylabel("Final Var(ψ)")
    axes[0].set_title("Final Variance vs D")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")

    axes[1].plot(D_values, final_Hs, "s-", lw=2, color="green")
    axes[1].set_xlabel("Diffusion coefficient D")
    axes[1].set_ylabel("Final H(ψ)")
    axes[1].set_title("Final Entropy vs D")
    axes[1].set_xscale("log")

    axes[2].plot(D_values, viable_fracs, "^-", lw=2, color="purple")
    axes[2].set_xlabel("Diffusion coefficient D")
    axes[2].set_ylabel("Viable fraction")
    axes[2].set_title("Viable Fraction vs D")
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


# ============================================================
# Batch utilities
# ============================================================

def run_batch_seeds(cfg: V2Phase1Config, seeds: list) -> Dict[str, Any]:
    """Run distributed simulation over multiple seeds."""
    results = []
    for s in seeds:
        c = V2Phase1Config(**{**cfg.__dict__, "seed": s})
        results.append(run_distributed(c))

    return {
        "mean_final_var": float(np.mean([r["final_var"] for r in results])),
        "std_final_var": float(np.std([r["final_var"] for r in results])),
        "mean_final_H": float(np.mean([r["final_H"] for r in results])),
        "std_final_H": float(np.std([r["final_H"] for r in results])),
        "mean_viable_fraction": float(np.mean([r["viable_fraction"] for r in results])),
        "std_viable_fraction": float(np.std([r["viable_fraction"] for r in results])),
        "n_seeds": len(seeds),
    }


# ============================================================
# Main — Phase 1 tests
# ============================================================

if __name__ == "__main__":

    import os
    os.makedirs("figures_v2", exist_ok=True)

    print("=" * 70)
    print("QMC Toy Model V2 — Phase 1: Single Node Proof of Concept")
    print("=" * 70)

    # --------------------------------------------------------
    # Test 1: Distributed vs Punctual
    # --------------------------------------------------------
    print("\n--- Test 1: Distributed vs Punctual ---")

    cfg_dist = V2Phase1Config(init_mode="gaussian", init_width=0.05, D=0.02)
    cfg_punct = V2Phase1Config(init_mode="narrow", D=0.02)

    result_dist = run_distributed(cfg_dist)
    result_punct = run_punctual(cfg_punct)

    plot_phase1_comparison(
        result_dist, result_punct,
        title="QMC V2 Phase 1 — Distributed vs Punctual (D=0.02)",
        save_path="figures_v2/phase1_test1_comparison.png"
    )

    print(f"  Distributed: final_Var={result_dist['final_var']:.6f}, "
          f"final_H={result_dist['final_H']:.4f}, "
          f"final_η_H={result_dist['final_eta_H']:.4f}, "
          f"viable_fraction={result_dist['viable_fraction']:.3f}")
    print(f"  Punctual:    final_τ={result_punct['final_tau']:.6f}, "
          f"H=0 always, viable_fraction={result_punct['viable_fraction']:.3f}")

    # Robustness over seeds
    print("\n  Robustness (20 seeds):")
    batch = run_batch_seeds(cfg_dist, list(range(20)))
    print(f"    mean_final_Var = {batch['mean_final_var']:.6f} ± {batch['std_final_var']:.6f}")
    print(f"    mean_final_H   = {batch['mean_final_H']:.4f} ± {batch['std_final_H']:.4f}")
    print(f"    mean_viable_fr = {batch['mean_viable_fraction']:.3f} ± {batch['std_viable_fraction']:.3f}")

    # --------------------------------------------------------
    # Test 2: Variance Corridor (D sweep)
    # --------------------------------------------------------
    print("\n--- Test 2: Variance Corridor (D sweep) ---")

    D_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    sweep_results = []

    for D_val in D_values:
        cfg_sweep = V2Phase1Config(init_mode="gaussian", init_width=0.05, D=D_val)
        res = run_distributed(cfg_sweep)
        sweep_results.append({
            "D": D_val,
            "final_var": res["final_var"],
            "final_H": res["final_H"],
            "final_eta_H": res["final_eta_H"],
            "viable_fraction": res["viable_fraction"],
        })
        print(f"  D={D_val:.3f} | final_Var={res['final_var']:.6f} | "
              f"final_H={res['final_H']:.4f} | η_H={res['final_eta_H']:.4f} | "
              f"viable={res['viable_fraction']:.3f}")

    plot_variance_sweep(
        sweep_results,
        title="QMC V2 Phase 1 — Variance Corridor (D sweep)",
        save_path="figures_v2/phase1_test2_D_sweep.png"
    )

    # --------------------------------------------------------
    # Test 2b: Cold start (low initial width)
    # --------------------------------------------------------
    print("\n--- Test 2b: Cold Start (narrow initial → thermalisation) ---")

    cfg_cold = V2Phase1Config(init_mode="narrow", D=0.02)
    result_cold = run_distributed(cfg_cold)

    print(f"  Cold start: final_Var={result_cold['final_var']:.6f}, "
          f"final_H={result_cold['final_H']:.4f}, "
          f"viable_fraction={result_cold['viable_fraction']:.3f}")

    # Compare cold start with warm start
    cfg_warm = V2Phase1Config(init_mode="gaussian", init_width=0.05, D=0.02)
    result_warm = run_distributed(cfg_warm)

    print(f"  Warm start: final_Var={result_warm['final_var']:.6f}, "
          f"final_H={result_warm['final_H']:.4f}, "
          f"viable_fraction={result_warm['viable_fraction']:.3f}")

    # Plot cold start
    plot_phase1_comparison(
        result_cold, result_punct,
        title="QMC V2 Phase 1 — Cold Start (near-punctual initial → thermalisation)",
        save_path="figures_v2/phase1_test2b_cold_start.png"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (Distributed vs Punctual):")
    print(f"    Punctual H = 0 always → entropically non-viable")
    print(f"    Distributed maintains H > 0 and Var > 0 → entropically viable")
    print(f"    Result: {'PASS' if result_dist['viable_fraction'] > 0.9 else 'INVESTIGATE'}")
    print(f"  Test 2 (Variance Corridor):")
    viable_D = [r for r in sweep_results if r["viable_fraction"] > 0.9]
    print(f"    Viable D range: {[r['D'] for r in viable_D] if viable_D else 'NONE'}")
    print(f"    Corridor exists: {'YES' if len(viable_D) >= 2 else 'NO'}")
    print(f"  Test 2b (Cold Start):")
    print(f"    Thermalisation: {'YES' if result_cold['final_var'] > cfg_cold.var_floor else 'NO'}")
    converged = abs(result_cold['final_var'] - result_warm['final_var']) / max(result_warm['final_var'], 1e-10)
    print(f"    Cold/Warm convergence: relative diff = {converged:.4f}")
    print("=" * 70)
