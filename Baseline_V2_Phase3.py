"""
QMC Toy Model V2 — Phase 3: Vectorial Field (2D Configurational Space)
========================================================================

Each node carries a distribution ψᵢ(t, θ₁, θ₂) over a 2D configurational
space Θ = Θ₁ × Θ₂ (M×M grid). Produces vectorial contributions to τ′ ∈ ℝ².

Tests:
  6. Vectorial vs scalar: does 2D Θ produce richer dynamics?
  7. Directional anisotropy: do the distributions develop directional structure?
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple


# ============================================================
# Configuration
# ============================================================

@dataclass
class V2Phase3Config:
    # Time
    T: int = 400
    dt: float = 0.04

    # Graph
    N: int = 8                     # fewer nodes (2D is heavier)
    ws_k: int = 4
    ws_p: float = 0.3

    # 2D configurational space
    M: int = 20                    # grid size per dimension (M×M total)
    theta_min: float = 0.0
    theta_max: float = 1.0

    # Dynamics
    D: float = 0.015
    V_strength: float = 1.2
    V_var_min: float = 0.002       # per-axis variance threshold
    V_var_max: float = 0.03
    coupling_strength: float = 0.06
    enable_memory: bool = True
    memory_rate: float = 0.008

    # Initial distributions
    init_spread: float = 0.25      # spread of anchor centres
    init_width: float = 0.04       # isotropic Gaussian width

    # Viability
    var_floor: float = 0.0005
    H_floor: float = 0.1
    delta_crit: float = 0.32
    g_min: float = 0.03

    # Seed
    seed: int = 42


# ============================================================
# 2D configurational space
# ============================================================

def make_theta_2d(cfg: V2Phase3Config) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (theta1_grid, theta2_grid, dtheta) for M×M grid."""
    t1 = np.linspace(cfg.theta_min, cfg.theta_max, cfg.M)
    t2 = np.linspace(cfg.theta_min, cfg.theta_max, cfg.M)
    dt = (cfg.theta_max - cfg.theta_min) / (cfg.M - 1)
    return t1, t2, dt


def make_meshgrid(t1: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(t1, t2, indexing='ij')


# ============================================================
# Initialisation
# ============================================================

def init_anchors_2d(cfg: V2Phase3Config) -> np.ndarray:
    """
    Distribute N anchor points in 2D Θ.
    Uses a circle layout for uniform angular separation.
    Returns: (N, 2) array of anchor centres.
    """
    angles = np.linspace(0, 2*np.pi, cfg.N, endpoint=False)
    radius = cfg.init_spread
    centers = np.column_stack([
        0.5 + radius * np.cos(angles),
        0.5 + radius * np.sin(angles)
    ])
    return np.clip(centers, 0.1, 0.9)


def init_psi_2d(cfg: V2Phase3Config, T1: np.ndarray, T2: np.ndarray,
                anchors: np.ndarray, dx: float) -> np.ndarray:
    """
    Initialise N distributions over M×M grid.
    Each is an isotropic Gaussian centered at its anchor.
    Returns: (N, M, M) array.
    """
    psi = np.zeros((cfg.N, cfg.M, cfg.M))
    for i in range(cfg.N):
        c1, c2 = anchors[i]
        g = np.exp(-0.5 * (((T1 - c1)/cfg.init_width)**2 +
                           ((T2 - c2)/cfg.init_width)**2))
        total = np.sum(g) * dx * dx
        psi[i] = g / total if total > 0 else np.ones((cfg.M, cfg.M)) / (cfg.M * cfg.M * dx * dx)
    return psi


def init_metric_2d(cfg: V2Phase3Config) -> np.ndarray:
    """Metric h(θ₁,θ₂) per node. Shape: (N, M, M)."""
    return np.ones((cfg.N, cfg.M, cfg.M))


# ============================================================
# 2D Observables
# ============================================================

def obs_expectation_2d(psi: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                        dx: float) -> np.ndarray:
    """Returns [𝔼[θ₁], 𝔼[θ₂]]."""
    dA = dx * dx
    mu1 = float(np.sum(T1 * psi * dA))
    mu2 = float(np.sum(T2 * psi * dA))
    return np.array([mu1, mu2])


def obs_variance_2d(psi: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                     dx: float) -> Tuple[float, float]:
    """Returns (Var₁, Var₂) — per-axis variances."""
    dA = dx * dx
    mu = obs_expectation_2d(psi, T1, T2, dx)
    v1 = float(np.sum((T1 - mu[0])**2 * psi * dA))
    v2 = float(np.sum((T2 - mu[1])**2 * psi * dA))
    return v1, v2


def obs_total_variance(psi: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                        dx: float) -> float:
    v1, v2 = obs_variance_2d(psi, T1, T2, dx)
    return v1 + v2


def obs_entropy_2d(psi: np.ndarray, dx: float) -> float:
    dA = dx * dx
    p = np.clip(psi * dA, 1e-30, None)
    return float(-np.sum(p * np.log(p)))


def obs_overlap_2d(psi_a: np.ndarray, psi_b: np.ndarray, dx: float) -> float:
    dA = dx * dx
    dot = np.sum(psi_a * psi_b * dA)
    na = np.sqrt(np.sum(psi_a**2 * dA))
    nb = np.sqrt(np.sum(psi_b**2 * dA))
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(dot / (na * nb))


def obs_anisotropy(psi: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                    dx: float) -> float:
    """Anisotropy = |Var₁ - Var₂| / (Var₁ + Var₂). 0 = isotropic, 1 = fully anisotropic."""
    v1, v2 = obs_variance_2d(psi, T1, T2, dx)
    total = v1 + v2
    if total < 1e-15:
        return 0.0
    return abs(v1 - v2) / total


# ============================================================
# 2D Force (per-node anchored)
# ============================================================

def force_2d(psi: np.ndarray, T1: np.ndarray, T2: np.ndarray,
             dx: float, anchor: np.ndarray,
             cfg: V2Phase3Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (F₁, F₂) force fields, each of shape (M, M).
    Repulsion from collapse + dispersion, plus anchor-return.
    """
    v1, v2 = obs_variance_2d(psi, T1, T2, dx)
    mu = obs_expectation_2d(psi, T1, T2, dx)
    total_var = v1 + v2

    disp1 = T1 - anchor[0]
    disp2 = T2 - anchor[1]

    # Collapse proximity
    collapse = np.exp(-total_var / (2 * cfg.V_var_min)) if total_var > 0 else 1.0

    # Dispersion proximity
    if total_var < 2 * cfg.V_var_max:
        dispersion = 0.0
    else:
        dispersion = np.exp(-(2*cfg.V_var_max - total_var)**2 / (cfg.V_var_max**2))

    # Anchor return
    ar1 = 0.3 * (mu[0] - anchor[0]) * np.ones_like(T1)
    ar2 = 0.3 * (mu[1] - anchor[1]) * np.ones_like(T2)

    F1 = cfg.V_strength * (collapse * disp1 - dispersion * disp1) - ar1
    F2 = cfg.V_strength * (collapse * disp2 - dispersion * disp2) - ar2

    return F1, F2


# ============================================================
# 2D Coupling (perturbative, anti-fusion)
# ============================================================

def coupling_2d(psi_i: np.ndarray, psi_all: np.ndarray,
                neighbors: List[int], dx: float,
                cfg: V2Phase3Config) -> np.ndarray:
    """Same principle as 1D: transmit novel component."""
    coup = np.zeros_like(psi_i)
    dA = dx * dx
    for j in neighbors:
        R = obs_overlap_2d(psi_i, psi_all[j], dx)
        novelty = psi_all[j] - R * psi_i
        coup += cfg.coupling_strength * R * novelty
    if len(neighbors) > 0:
        coup /= len(neighbors)
    return coup


# ============================================================
# 2D Fokker-Planck step
# ============================================================

def fp_step_2d(psi: np.ndarray, T1: np.ndarray, T2: np.ndarray,
               dx: float, h: np.ndarray, coupling: np.ndarray,
               anchor: np.ndarray, cfg: V2Phase3Config,
               rng: np.random.Generator) -> np.ndarray:
    M = cfg.M

    # --- 2D Laplacian (central differences) ---
    lap = np.zeros((M, M))
    # Interior
    lap[1:-1, :] += (psi[2:, :] - 2*psi[1:-1, :] + psi[:-2, :]) / dx**2
    lap[:, 1:-1] += (psi[:, 2:] - 2*psi[:, 1:-1] + psi[:, :-2]) / dx**2
    # Reflecting boundaries
    lap[0, :] += (psi[1, :] - psi[0, :]) / dx**2
    lap[-1, :] += (psi[-2, :] - psi[-1, :]) / dx**2
    lap[:, 0] += (psi[:, 1] - psi[:, 0]) / dx**2
    lap[:, -1] += (psi[:, -2] - psi[:, -1]) / dx**2

    D_eff = cfg.D * h
    diffusion = D_eff * lap

    # --- 2D Drift (simplified: F·∇ψ approximation) ---
    F1, F2 = force_2d(psi, T1, T2, dx, anchor, cfg)

    # Gradient of ψ (central differences)
    dpsi_d1 = np.zeros((M, M))
    dpsi_d1[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2*dx)
    dpsi_d1[0, :] = (psi[1, :] - psi[0, :]) / dx
    dpsi_d1[-1, :] = (psi[-1, :] - psi[-2, :]) / dx

    dpsi_d2 = np.zeros((M, M))
    dpsi_d2[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2*dx)
    dpsi_d2[:, 0] = (psi[:, 1] - psi[:, 0]) / dx
    dpsi_d2[:, -1] = (psi[:, -1] - psi[:, -2]) / dx

    # Drift = -div(F·ψ) ≈ -(F·∇ψ + ψ·div(F))
    # Simplified to -F·∇ψ (conservative approximation)
    drift = -(F1 * dpsi_d1 + F2 * dpsi_d2)

    # --- Noise ---
    noise = rng.normal(0, cfg.D * 0.08, (M, M)) * psi

    # --- Update ---
    psi_new = psi + cfg.dt * (diffusion + drift + coupling + noise)
    psi_new = np.maximum(psi_new, 0.0)

    # Renormalise
    dA = dx * dx
    total = np.sum(psi_new) * dA
    if total > 0:
        psi_new /= total
    else:
        psi_new = np.ones((M, M)) / (M * M * dA)

    return psi_new


# ============================================================
# Main simulation
# ============================================================

def run_phase3(cfg: V2Phase3Config) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    t1, t2, dx = make_theta_2d(cfg)
    T1, T2 = make_meshgrid(t1, t2)

    G = nx.watts_strogatz_graph(cfg.N, k=cfg.ws_k, p=cfg.ws_p, seed=cfg.seed)
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        G = nx.convert_node_labels_to_integers(G)
    A = nx.to_numpy_array(G)
    neighbors = {i: list(G.neighbors(i)) for i in range(cfg.N)}

    anchors = init_anchors_2d(cfg)
    psi = init_psi_2d(cfg, T1, T2, anchors, dx)
    h = init_metric_2d(cfg)

    N = cfg.N
    # History
    var_history = np.zeros((cfg.T, N))          # total variance
    H_history = np.zeros((cfg.T, N))
    aniso_history = np.zeros((cfg.T, N))        # anisotropy
    expect_history = np.zeros((cfg.T, N, 2))    # 2D expectations
    delta_history = np.zeros(cfg.T)
    g_history = np.zeros(cfg.T)
    viable_history = np.zeros(cfg.T, dtype=bool)
    mean_R_history = np.zeros(cfg.T)

    # Snapshots
    psi_snapshots = []

    for t in range(cfg.T):
        # --- Observables ---
        for i in range(N):
            var_history[t, i] = obs_total_variance(psi[i], T1, T2, dx)
            H_history[t, i] = obs_entropy_2d(psi[i], dx)
            aniso_history[t, i] = obs_anisotropy(psi[i], T1, T2, dx)
            expect_history[t, i] = obs_expectation_2d(psi[i], T1, T2, dx)

        # Aggregate (from expectations component 1 — scalar projection)
        exps = expect_history[t, :, 0]  # project onto θ₁
        delta_history[t] = float(np.std(exps))
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if A[i, j] > 0:
                    edges.append(np.linalg.norm(expect_history[t, i] - expect_history[t, j]))
        g_history[t] = float(np.mean(edges)) if edges else 0.0
        viable_history[t] = (delta_history[t] > 0 and g_history[t] > cfg.g_min)

        # Mean overlap
        overlaps = []
        for i in range(N):
            for j in range(i+1, N):
                if A[i, j] > 0:
                    overlaps.append(obs_overlap_2d(psi[i], psi[j], dx))
        mean_R_history[t] = float(np.mean(overlaps)) if overlaps else 0.0

        # Snapshots
        if t in [0, cfg.T//4, cfg.T//2, 3*cfg.T//4, cfg.T-1]:
            psi_snapshots.append((t, psi.copy()))

        # --- Dynamics ---
        psi_new = np.zeros_like(psi)
        for i in range(N):
            coup = coupling_2d(psi[i], psi, neighbors[i], dx, cfg)
            psi_new[i] = fp_step_2d(psi[i], T1, T2, dx, h[i], coup,
                                     anchors[i], cfg, rng)
            if cfg.enable_memory:
                h[i] *= np.exp(-cfg.memory_rate * psi[i] * dx)
        psi = psi_new

    return {
        "config": cfg,
        "t1": t1, "t2": t2, "anchors": anchors,
        "var_history": var_history,
        "H_history": H_history,
        "aniso_history": aniso_history,
        "expect_history": expect_history,
        "delta_history": delta_history,
        "g_history": g_history,
        "viable_history": viable_history,
        "mean_R_history": mean_R_history,
        "psi_snapshots": psi_snapshots,
        "metric_final": h.copy(),
        "final_delta": float(delta_history[-1]),
        "final_g": float(g_history[-1]),
        "final_mean_R": float(mean_R_history[-1]),
        "final_mean_var": float(np.mean(var_history[-1])),
        "final_mean_H": float(np.mean(H_history[-1])),
        "final_mean_aniso": float(np.mean(aniso_history[-1])),
        "viable_fraction": float(np.mean(viable_history)),
    }


# ============================================================
# Plotting
# ============================================================

def plot_phase3(result: Dict, title: str = "QMC V2 Phase 3 — Vectorial (2D)",
                save_path: Optional[str] = None) -> None:
    cfg = result["config"]
    ts = np.arange(cfg.T)

    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(title, fontsize=15, y=0.99)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.38, wspace=0.32)

    # --- Row 1: Corridor, Gradient, Viability ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ts, result["delta_history"], lw=2, color="blue")
    ax1.axhline(cfg.delta_crit, ls=":", color="red", lw=1)
    ax1.set_title("Delta_agg")
    ax1.set_xlabel("t")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ts, result["g_history"], lw=2, color="darkorange")
    ax2.axhline(cfg.g_min, ls=":", color="red", lw=1)
    ax2.set_title("G_agg (2D norm)")
    ax2.set_xlabel("t")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ts, result["viable_history"].astype(int), lw=2, color="green")
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["No", "Yes"])
    ax3.set_title("Viable")
    ax3.set_xlabel("t")

    # --- Row 2: Var, H, Overlap ---
    ax4 = fig.add_subplot(gs[1, 0])
    for i in range(cfg.N):
        ax4.plot(ts, result["var_history"][:, i], lw=0.8, alpha=0.6)
    ax4.set_title("Var(psi) per node")
    ax4.set_xlabel("t")

    ax5 = fig.add_subplot(gs[1, 1])
    for i in range(cfg.N):
        ax5.plot(ts, result["H_history"][:, i], lw=0.8, alpha=0.6)
    ax5.set_title("H(psi) per node")
    ax5.set_xlabel("t")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(ts, result["mean_R_history"], lw=2, color="purple")
    ax6.axhline(0.2, ls=":", color="red", lw=1)
    ax6.axhline(0.5, ls=":", color="red", lw=1)
    ax6.set_ylim(-0.05, 1.05)
    ax6.set_title("Mean overlap R")
    ax6.set_xlabel("t")

    # --- Row 3: Anisotropy + 2D trajectories ---
    ax7 = fig.add_subplot(gs[2, 0])
    for i in range(cfg.N):
        ax7.plot(ts, result["aniso_history"][:, i], lw=0.8, alpha=0.6)
    ax7.set_title("Anisotropy per node")
    ax7.set_xlabel("t")
    ax7.set_ylim(-0.05, 1.05)

    ax8 = fig.add_subplot(gs[2, 1])
    for i in range(cfg.N):
        ax8.plot(result["expect_history"][:, i, 0],
                 result["expect_history"][:, i, 1],
                 lw=0.6, alpha=0.7)
    ax8.scatter(result["anchors"][:, 0], result["anchors"][:, 1],
                marker="x", s=80, color="red", zorder=5, label="anchors")
    ax8.set_title("2D trajectories E[psi]")
    ax8.set_xlabel("theta_1")
    ax8.set_ylabel("theta_2")
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.legend(fontsize=8)
    ax8.set_aspect("equal")

    # --- Row 3 col 3: empty or metric ---
    ax9 = fig.add_subplot(gs[2, 2])
    # Show final metric deformation for node 0
    h0 = result["metric_final"][0]
    im = ax9.imshow(h0.T, origin="lower", extent=[0, 1, 0, 1],
                     cmap="viridis", aspect="equal")
    ax9.set_title("Final metric h_0(theta)")
    plt.colorbar(im, ax=ax9, fraction=0.046)

    # --- Row 4: Distribution snapshots (heatmaps) ---
    snap_nodes = [0, cfg.N//2, cfg.N-1]
    for col, ni in enumerate(snap_nodes):
        ax = fig.add_subplot(gs[3, col])
        if result["psi_snapshots"]:
            t_snap, psi_snap = result["psi_snapshots"][-1]
            if ni < cfg.N:
                im = ax.imshow(psi_snap[ni].T, origin="lower",
                               extent=[0, 1, 0, 1], cmap="hot", aspect="equal")
                ax.scatter(*result["anchors"][ni], marker="x", s=60, color="cyan")
                ax.set_title(f"psi_{ni} at t={t_snap}")
                plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.makedirs("figures_v2", exist_ok=True)

    print("=" * 70)
    print("QMC Toy Model V2 — Phase 3: Vectorial (2D)")
    print("=" * 70)

    # --- Test 6: 2D viability ---
    print("\n--- Test 6: 2D distributional viability ---")
    cfg = V2Phase3Config()
    result = run_phase3(cfg)

    plot_phase3(result,
                title="QMC V2 Phase 3 — 2D Distributional Viability",
                save_path="figures_v2/phase3_test6_2d.png")

    print(f"  viable_fraction = {result['viable_fraction']:.3f}")
    print(f"  final_delta     = {result['final_delta']:.4f}")
    print(f"  final_g         = {result['final_g']:.4f}")
    print(f"  final_mean_R    = {result['final_mean_R']:.4f}")
    print(f"  final_mean_Var  = {result['final_mean_var']:.6f}")
    print(f"  final_mean_H    = {result['final_mean_H']:.4f}")
    print(f"  final_mean_aniso= {result['final_mean_aniso']:.4f}")

    # Robustness
    print("\n  Robustness (5 seeds):")
    vfs = []
    for s in range(5):
        c = V2Phase3Config(seed=s)
        r = run_phase3(c)
        vfs.append(r["viable_fraction"])
    print(f"    viable: {np.mean(vfs):.3f} +/- {np.std(vfs):.3f}")

    # --- Test 7: Directional anisotropy ---
    print("\n--- Test 7: Directional anisotropy ---")
    # Run with asymmetric initial width
    cfg_aniso = V2Phase3Config(seed=42)
    result_aniso = run_phase3(cfg_aniso)
    mean_aniso = np.mean(result_aniso["aniso_history"][-50:])
    print(f"  Mean anisotropy (last 50 steps): {mean_aniso:.4f}")
    print(f"  Anisotropy > 0 (distributions develop directional structure): "
          f"{'YES' if mean_aniso > 0.01 else 'MINIMAL'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY")
    print("=" * 70)
    print(f"  Test 6 (2D viability): viable_fraction = {result['viable_fraction']:.3f}")
    print(f"  Test 7 (Anisotropy):   mean = {mean_aniso:.4f}")
    print(f"  Anti-fusion:           R = {result['final_mean_R']:.4f} "
          f"({'OK' if 0.1 <= result['final_mean_R'] <= 0.6 else 'CHECK'})")
    print("=" * 70)
