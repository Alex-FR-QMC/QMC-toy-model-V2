"""
QMC Toy Model V2 — Phase 2: Graph and Inter-nodal Coupling
============================================================

N nodes on a Watts-Strogatz graph, each carrying a distribution ψᵢ(t)
over M discrete bins. Coupling is perturbative — it transmits the novel
component of neighbor distributions, not an attractive force.

Anti-fusion by design: coupling term ∝ Rᵢⱼ · (ψⱼ − Rᵢⱼ · ψᵢ)
- When Rᵢⱼ → 1 (identical): coupling → 0 (nothing new to transmit)
- When Rᵢⱼ → 0 (decoupled): coupling → 0 (no channel)
- Maximum at intermediate overlap: perturbation, not attraction

Tests:
  3. Necessity of 𝒩: with vs without V_𝒩
  4. Distributed coupling vs V1 scalar coupling
  5. Distributional memory: δ𝓗^dist present vs absent
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple


# ============================================================
# Configuration
# ============================================================

@dataclass
class V2Phase2Config:
    # Time
    T: int = 500
    dt: float = 0.05

    # Graph
    N: int = 12                    # number of nodes
    graph_mode: str = "ws"
    ws_k: int = 4
    ws_p: float = 0.3

    # Configurational space
    M: int = 40                    # bins per node (increased for resolution)
    theta_min: float = 0.0
    theta_max: float = 1.0

    # Dynamics
    D: float = 0.02                # diffusion (𝒯)
    V_strength: float = 1.0        # repulsive potential (𝒩)
    V_var_min: float = 0.003       # KNV 5 threshold
    V_var_max: float = 0.04        # KNV 6 threshold (tighter for better confinement)
    coupling_strength: float = 0.08  # inter-nodal coupling (reduced to prevent fusion)
    enable_V: bool = True          # toggle V_𝒩 (for test 3)
    enable_memory: bool = True     # toggle metric deformation (for test 5)
    memory_rate: float = 0.01      # rate of metric deformation

    # Initial distributions
    init_spread: float = 0.30      # spread of initial centers (wider separation)
    init_width: float = 0.03       # width of each node's initial Gaussian (narrower)

    # Viability thresholds
    var_floor: float = 0.001
    H_floor: float = 0.05
    delta_crit: float = 0.32       # Chapter 1 corridor
    g_min: float = 0.045           # Chapter 1 gradient

    # Analysis
    lethargy_window: int = 30
    overlap_window: int = 20       # window for A_H computation

    # Seed
    seed: int = 42


# ============================================================
# Graph
# ============================================================

def make_graph(cfg: V2Phase2Config) -> nx.Graph:
    G = nx.watts_strogatz_graph(cfg.N, k=cfg.ws_k, p=cfg.ws_p, seed=cfg.seed)
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


# ============================================================
# Configurational space
# ============================================================

def make_theta(cfg: V2Phase2Config) -> np.ndarray:
    return np.linspace(cfg.theta_min, cfg.theta_max, cfg.M)


def dtheta(cfg: V2Phase2Config) -> float:
    return (cfg.theta_max - cfg.theta_min) / (cfg.M - 1)


# ============================================================
# Initialisation
# ============================================================

def init_distributions(cfg: V2Phase2Config, theta: np.ndarray,
                        rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialise N distributions over M bins.
    Each node gets a Gaussian centered at a different position.
    Returns: (psi array of shape (N, M), anchor_centers array of shape (N,))
    """
    dt = dtheta(cfg)
    centers = np.linspace(0.5 - cfg.init_spread, 0.5 + cfg.init_spread, cfg.N)
    psi = np.zeros((cfg.N, cfg.M))

    for i in range(cfg.N):
        psi[i] = np.exp(-0.5 * ((theta - centers[i]) / cfg.init_width)**2)
        psi[i] /= np.sum(psi[i]) * dt

    return psi, centers


def init_metric(cfg: V2Phase2Config) -> np.ndarray:
    """
    Initialise deformed metric hᵢ(θ) for each node.
    Starts uniform (h = 1 everywhere). Shape: (N, M).
    """
    return np.ones((cfg.N, cfg.M))


# ============================================================
# Observables
# ============================================================

def compute_var(psi: np.ndarray, theta: np.ndarray, dt: float) -> float:
    mu = np.sum(theta * psi * dt)
    return float(np.sum((theta - mu)**2 * psi * dt))


def compute_entropy(psi: np.ndarray, dt: float) -> float:
    p = np.clip(psi * dt, 1e-30, None)
    return float(-np.sum(p * np.log(p)))


def compute_expectation(psi: np.ndarray, theta: np.ndarray, dt: float) -> float:
    return float(np.sum(theta * psi * dt))


def compute_overlap(psi_a: np.ndarray, psi_b: np.ndarray, dt: float) -> float:
    """Rₐ,ᵦ = ∫ψₐψᵦ / (‖ψₐ‖‖ψᵦ‖)"""
    dot = np.sum(psi_a * psi_b * dt)
    norm_a = np.sqrt(np.sum(psi_a**2 * dt))
    norm_b = np.sqrt(np.sum(psi_b**2 * dt))
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_aggregate_delta(psi_all: np.ndarray, theta: np.ndarray, dt: float) -> float:
    """Δ_agg = std of the expectations (scalar projection analogue)."""
    expectations = np.array([compute_expectation(psi_all[i], theta, dt)
                             for i in range(psi_all.shape[0])])
    return float(np.std(expectations))


def compute_aggregate_g(psi_all: np.ndarray, theta: np.ndarray, dt: float,
                         A: np.ndarray) -> float:
    """𝒢_agg = mean |𝔼[ψᵢ] - 𝔼[ψⱼ]| over edges."""
    N = psi_all.shape[0]
    expectations = np.array([compute_expectation(psi_all[i], theta, dt)
                             for i in range(N)])
    edge_diffs = []
    for i in range(N):
        for j in range(i+1, N):
            if A[i, j] > 0:
                edge_diffs.append(abs(expectations[i] - expectations[j]))
    return float(np.mean(edge_diffs)) if edge_diffs else 0.0


def compute_T_H(H: float, omega_var: float, R_productive: float) -> float:
    """𝒯_H = H · ω^var · Σ Rᵢⱼ(1-Rᵢⱼ)"""
    return H * omega_var * R_productive


# ============================================================
# Repulsive potential (same as Phase 1)
# ============================================================

def compute_force(theta: np.ndarray, psi: np.ndarray, dt: float,
                  cfg: V2Phase2Config, anchor: float = 0.5) -> np.ndarray:
    if not cfg.enable_V:
        return np.zeros_like(theta)

    var = compute_var(psi, theta, dt)
    mu = compute_expectation(psi, theta, dt)

    # Displacement from the node's own anchor (not global center)
    displacement = theta - anchor

    # Collapse proximity: how close Var is to 0
    collapse_prox = np.exp(-var / cfg.V_var_min) if var > 0 else 1.0

    # Dispersion proximity: how close Var is to Var_crit
    if var < cfg.V_var_max:
        dispersion_prox = 0.0
    else:
        dispersion_prox = np.exp(-(cfg.V_var_max - var)**2 / (0.5 * cfg.V_var_max**2))

    # Anchor-return force: pulls expectation back toward anchor
    anchor_return = 0.3 * (mu - anchor) * np.ones_like(theta)

    F = cfg.V_strength * (
        collapse_prox * displacement
        - dispersion_prox * displacement
    ) - anchor_return

    return F


# ============================================================
# Inter-nodal coupling (perturbative, anti-fusion)
# ============================================================

def compute_coupling(psi_i: np.ndarray, psi_all: np.ndarray,
                     neighbors: List[int], dt: float,
                     cfg: V2Phase2Config) -> np.ndarray:
    """
    Perturbative coupling: transmits the NOVEL component of neighbor distributions.

    coupling_ij(θ) = κ · Rᵢⱼ · (ψⱼ(θ) − Rᵢⱼ · ψᵢ(θ))

    Properties:
    - When Rᵢⱼ → 1 (identical): (ψⱼ − 1·ψᵢ) → 0 → no coupling
    - When Rᵢⱼ → 0 (decoupled): Rᵢⱼ·(...) → 0 → no coupling
    - Maximum influence at intermediate overlap: perturbation, not attraction
    - Transmits what j has that i doesn't (novelty), weighted by shared presence (overlap)
    """
    coupling = np.zeros_like(psi_i)

    for j in neighbors:
        R_ij = compute_overlap(psi_i, psi_all[j], dt)
        # Novel component of ψⱼ relative to ψᵢ
        novelty = psi_all[j] - R_ij * psi_i
        # Coupling: overlap-weighted novelty
        coupling += cfg.coupling_strength * R_ij * novelty

    if len(neighbors) > 0:
        coupling /= len(neighbors)

    return coupling


# ============================================================
# Metric deformation (morphological memory)
# ============================================================

def update_metric(h: np.ndarray, psi: np.ndarray, dt_val: float,
                  cfg: V2Phase2Config) -> np.ndarray:
    """
    hᵢ(θ) evolves: regions of high ψᵢ get metrically contracted.
    h(t+1) = h(t) · exp(-β · ψ · dt)
    """
    if not cfg.enable_memory:
        return h
    return h * np.exp(-cfg.memory_rate * psi * dt_val)


# ============================================================
# Fokker-Planck step (single node, with coupling)
# ============================================================

def fp_step(psi_i: np.ndarray, theta: np.ndarray, dt_val: float,
            h_i: np.ndarray, coupling_term: np.ndarray,
            cfg: V2Phase2Config, rng: np.random.Generator,
            anchor: float = 0.5) -> np.ndarray:
    M = len(psi_i)
    dx = dtheta(cfg)

    # Diffusion: D · Δ_θ ψ (central differences)
    lap = np.zeros(M)
    lap[1:-1] = (psi_i[2:] - 2*psi_i[1:-1] + psi_i[:-2]) / dx**2
    lap[0] = (psi_i[1] - psi_i[0]) / dx**2
    lap[-1] = (psi_i[-2] - psi_i[-1]) / dx**2

    # Effective diffusion modulated by metric (wider where h is large)
    D_eff = cfg.D * h_i
    diffusion = D_eff * lap

    # Drift: -∂_θ[F·ψ] (upwind)
    F = compute_force(theta, psi_i, dx, cfg, anchor=anchor)
    flux = F * psi_i
    drift = np.zeros(M)
    for k in range(1, M-1):
        if F[k] >= 0:
            drift[k] = -(flux[k] - flux[k-1]) / dx
        else:
            drift[k] = -(flux[k+1] - flux[k]) / dx

    # Granular noise
    noise = rng.normal(0, cfg.D * 0.1, M) * psi_i

    # Update
    psi_new = psi_i + cfg.dt * (diffusion + drift + coupling_term + noise)
    psi_new = np.maximum(psi_new, 0.0)

    # Renormalise
    total = np.sum(psi_new) * dx
    if total > 0:
        psi_new /= total
    else:
        psi_new = np.ones(M) / (cfg.theta_max - cfg.theta_min)

    return psi_new


# ============================================================
# Main simulation
# ============================================================

def run_phase2(cfg: V2Phase2Config) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    theta = make_theta(cfg)
    dx = dtheta(cfg)

    G = make_graph(cfg)
    A = nx.to_numpy_array(G)
    N = cfg.N

    # Neighbour lists
    neighbors = {i: list(G.neighbors(i)) for i in range(N)}

    # Initialise
    psi, anchors = init_distributions(cfg, theta, rng)
    h = init_metric(cfg)

    # History
    var_history = np.zeros((cfg.T, N))
    H_history = np.zeros((cfg.T, N))
    expect_history = np.zeros((cfg.T, N))
    delta_history = np.zeros(cfg.T)
    g_history = np.zeros(cfg.T)
    viable_history = np.zeros(cfg.T, dtype=bool)

    # Inter-nodal observables
    overlap_history = []  # list of (T,) arrays, one per pair tracked
    mean_overlap_history = np.zeros(cfg.T)
    A_H_history = np.zeros(cfg.T)
    T_H_history = np.zeros(cfg.T)
    dephasing_history = np.zeros(cfg.T)

    # Track a few pairs for detailed overlap
    tracked_pairs = [(0, 1), (0, N//2), (N//4, 3*N//4)]
    pair_overlap_history = {p: np.zeros(cfg.T) for p in tracked_pairs}

    # Distribution snapshots
    psi_snapshots = []

    for t in range(cfg.T):
        # --- Per-node observables ---
        for i in range(N):
            var_history[t, i] = compute_var(psi[i], theta, dx)
            H_history[t, i] = compute_entropy(psi[i], dx)
            expect_history[t, i] = compute_expectation(psi[i], theta, dx)

        # --- Aggregate observables ---
        delta_history[t] = compute_aggregate_delta(psi, theta, dx)
        g_history[t] = compute_aggregate_g(psi, theta, dx, A)
        viable_history[t] = (0 < delta_history[t] < cfg.delta_crit) and (g_history[t] > cfg.g_min)

        # --- Inter-nodal observables ---
        overlaps_t = []
        for i in range(N):
            for j in range(i+1, N):
                if A[i, j] > 0:
                    R = compute_overlap(psi[i], psi[j], dx)
                    overlaps_t.append(R)
        mean_R = np.mean(overlaps_t) if overlaps_t else 0.0
        mean_overlap_history[t] = mean_R

        # Tracked pairs
        for p in tracked_pairs:
            if p[0] < N and p[1] < N:
                pair_overlap_history[p][t] = compute_overlap(psi[p[0]], psi[p[1]], dx)

        # A_H: variance of overlap over window
        if t >= cfg.overlap_window:
            A_H_history[t] = np.var(mean_overlap_history[t-cfg.overlap_window:t])
        else:
            A_H_history[t] = 0.0

        # Productive overlap: Σ R(1-R)
        R_productive = sum(R * (1 - R) for R in overlaps_t) if overlaps_t else 0.0

        # Dephasing: Var of expectations
        dephasing_history[t] = np.var(expect_history[t])

        # 𝒯_H (system-level)
        mean_H = np.mean(H_history[t])
        omega_var = np.mean(np.abs(np.diff(var_history[max(0,t-1):t+1], axis=0))) if t > 0 else 0.0
        T_H_history[t] = compute_T_H(mean_H, omega_var, R_productive)

        # Snapshots
        if t in [0, cfg.T//4, cfg.T//2, 3*cfg.T//4, cfg.T-1]:
            psi_snapshots.append((t, psi.copy()))

        # --- Dynamics ---
        psi_new = np.zeros_like(psi)
        for i in range(N):
            coup = compute_coupling(psi[i], psi, neighbors[i], dx, cfg)
            psi_new[i] = fp_step(psi[i], theta, dx, h[i], coup, cfg, rng,
                                  anchor=anchors[i])

            # Update metric
            h[i] = update_metric(h[i], psi[i], dx, cfg)

        psi = psi_new

    return {
        "config": cfg,
        "theta": theta,
        "graph": G,
        "adjacency": A,
        "var_history": var_history,
        "H_history": H_history,
        "expect_history": expect_history,
        "delta_history": delta_history,
        "g_history": g_history,
        "viable_history": viable_history,
        "mean_overlap_history": mean_overlap_history,
        "pair_overlap_history": pair_overlap_history,
        "A_H_history": A_H_history,
        "T_H_history": T_H_history,
        "dephasing_history": dephasing_history,
        "psi_snapshots": psi_snapshots,
        "metric_final": h.copy(),
        "final_delta": float(delta_history[-1]),
        "final_g": float(g_history[-1]),
        "final_mean_overlap": float(mean_overlap_history[-1]),
        "final_mean_var": float(np.mean(var_history[-1])),
        "final_mean_H": float(np.mean(H_history[-1])),
        "viable_fraction": float(np.mean(viable_history)),
    }


# ============================================================
# V1 comparison (scalar, same graph)
# ============================================================

def run_v1_comparison(cfg: V2Phase2Config) -> Dict[str, Any]:
    """V1-like scalar dynamics on the same graph for comparison."""
    rng = np.random.default_rng(cfg.seed)
    G = make_graph(cfg)
    A = nx.to_numpy_array(G)
    N = cfg.N

    tau = np.linspace(0.5 - cfg.init_spread, 0.5 + cfg.init_spread, N)
    tau_history = np.zeros((cfg.T, N))
    delta_history = np.zeros(cfg.T)
    g_history = np.zeros(cfg.T)
    viable_history = np.zeros(cfg.T, dtype=bool)

    for t in range(cfg.T):
        tau_history[t] = tau.copy()
        delta_history[t] = float(np.std(tau))

        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if A[i, j] > 0:
                    edges.append(abs(tau[i] - tau[j]))
        g_history[t] = float(np.mean(edges)) if edges else 0.0
        viable_history[t] = (0 < delta_history[t] < cfg.delta_crit) and (g_history[t] > cfg.g_min)

        # V1 dynamics
        degree = np.sum(A, axis=1)
        coupling = cfg.coupling_strength * (A @ tau - degree * tau)
        reflexive = -0.15 * (tau - np.mean(tau))
        noise = rng.normal(0, 0.028, N)
        dtau = cfg.dt * (coupling + reflexive + 0.48 * noise)
        tau = np.clip(tau + dtau, 0.09, 0.91)

    return {
        "config": cfg,
        "tau_history": tau_history,
        "delta_history": delta_history,
        "g_history": g_history,
        "viable_history": viable_history,
        "final_delta": float(delta_history[-1]),
        "final_g": float(g_history[-1]),
        "viable_fraction": float(np.mean(viable_history)),
    }


# ============================================================
# Plotting
# ============================================================

def plot_phase2(result: Dict, v1_result: Optional[Dict] = None,
                title: str = "QMC V2 Phase 2",
                save_path: Optional[str] = None) -> None:
    cfg = result["config"]
    T = cfg.T
    ts = np.arange(T)

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(title, fontsize=16, y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.38, wspace=0.28)

    # --- Panel 1: Expectations (all nodes) ---
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(cfg.N):
        ax1.plot(ts, result["expect_history"][:, i], lw=0.8, alpha=0.7)
    if v1_result is not None:
        for i in range(cfg.N):
            ax1.plot(ts, v1_result["tau_history"][:, i], lw=0.5, alpha=0.3, color="gray")
    ax1.set_title("Node expectations 𝔼[ψᵢ] (color) vs V1 τᵢ (gray)")
    ax1.set_xlabel("Time steps")

    # --- Panel 2: Δ and 𝒢 ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ts, result["delta_history"], lw=2, label="Δ_agg (V2)", color="blue")
    ax2.plot(ts, result["g_history"], lw=2, label="𝒢_agg (V2)", color="darkorange")
    if v1_result is not None:
        ax2.plot(ts, v1_result["delta_history"], lw=1.5, ls="--", label="Δ (V1)", color="lightblue")
        ax2.plot(ts, v1_result["g_history"], lw=1.5, ls="--", label="𝒢 (V1)", color="lightsalmon")
    ax2.axhline(cfg.delta_crit, ls=":", color="red", lw=1, label="Δ_crit")
    ax2.axhline(cfg.g_min, ls=":", color="green", lw=1, label="𝒢_min")
    ax2.set_title("Morphodynamic Corridor & Gradient")
    ax2.set_xlabel("Time steps")
    ax2.legend(fontsize=7)

    # --- Panel 3: Mean Var and H ---
    ax3 = fig.add_subplot(gs[1, 0])
    mean_var = np.mean(result["var_history"], axis=1)
    mean_H = np.mean(result["H_history"], axis=1)
    ax3.plot(ts, mean_var, lw=2, label="mean Var(ψ)", color="blue")
    ax3.set_ylabel("Var", color="blue")
    ax3b = ax3.twinx()
    ax3b.plot(ts, mean_H, lw=2, label="mean H(ψ)", color="green")
    ax3b.set_ylabel("H", color="green")
    ax3.set_title("Mean Variance and Entropy across nodes")
    ax3.set_xlabel("Time steps")

    # --- Panel 4: Inter-nodal overlap ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(ts, result["mean_overlap_history"], lw=2, label="mean R", color="purple")
    for p, hist in result["pair_overlap_history"].items():
        ax4.plot(ts, hist, lw=0.8, alpha=0.5, label=f"R_{p}")
    ax4.axhline(0.2, ls=":", color="red", lw=1)
    ax4.axhline(0.5, ls=":", color="red", lw=1)
    ax4.set_title("Inter-nodal overlap Rₐ,ᵦ")
    ax4.set_xlabel("Time steps")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(fontsize=7)

    # --- Panel 5: A_H and 𝒯_H ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(ts, result["A_H_history"], lw=2, label="A_H", color="red")
    ax5.set_ylabel("A_H", color="red")
    ax5b = ax5.twinx()
    ax5b.plot(ts, result["T_H_history"], lw=2, label="𝒯_H", color="teal")
    ax5b.set_ylabel("𝒯_H", color="teal")
    ax5.set_title("Anti-coherence derivative A_H and Exploratory tendency 𝒯_H")
    ax5.set_xlabel("Time steps")

    # --- Panel 6: Dephasing ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(ts, result["dephasing_history"], lw=2, color="brown")
    ax6.set_title("Dephasing Var(𝔼[ψᵢ])")
    ax6.set_xlabel("Time steps")

    # --- Panel 7: Viability ---
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(ts, result["viable_history"].astype(int), lw=2, color="green", label="V2")
    if v1_result is not None:
        ax7.plot(ts, v1_result["viable_history"].astype(int), lw=1.5, ls="--",
                 color="gray", label="V1")
    ax7.set_ylim(-0.1, 1.1)
    ax7.set_yticks([0, 1])
    ax7.set_yticklabels(["No", "Yes"])
    ax7.set_title("Membership in 𝒱")
    ax7.set_xlabel("Time steps")
    ax7.legend()

    # --- Panel 8: Distribution snapshots (selected nodes) ---
    ax8 = fig.add_subplot(gs[3, 1])
    theta = result["theta"]
    if result["psi_snapshots"]:
        t_snap, psi_snap = result["psi_snapshots"][-1]  # final snapshot
        for i in [0, cfg.N//4, cfg.N//2, 3*cfg.N//4, cfg.N-1]:
            if i < cfg.N:
                ax8.plot(theta, psi_snap[i], label=f"node {i}", lw=1.5)
    ax8.set_title(f"Final distributions ψᵢ(θ) at t={cfg.T-1}")
    ax8.set_xlabel("θ")
    ax8.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Main — Phase 2 tests
# ============================================================

if __name__ == "__main__":
    os.makedirs("figures_v2", exist_ok=True)

    print("=" * 70)
    print("QMC Toy Model V2 — Phase 2: Graph and Coupling")
    print("=" * 70)

    # --------------------------------------------------------
    # Test 3: Distributed coupled vs V1 scalar coupled
    # --------------------------------------------------------
    print("\n--- Test 3: Distributed coupled vs V1 scalar ---")

    cfg = V2Phase2Config()
    result = run_phase2(cfg)
    v1_result = run_v1_comparison(cfg)

    plot_phase2(result, v1_result,
                title="QMC V2 Phase 2 — Distributed coupled vs V1 scalar",
                save_path="figures_v2/phase2_test3_comparison.png")

    print(f"  V2: final_Δ={result['final_delta']:.4f}, final_𝒢={result['final_g']:.4f}, "
          f"viable_fraction={result['viable_fraction']:.3f}")
    print(f"      final_mean_R={result['final_mean_overlap']:.4f}, "
          f"final_mean_Var={result['final_mean_var']:.6f}, "
          f"final_mean_H={result['final_mean_H']:.4f}")
    print(f"  V1: final_Δ={v1_result['final_delta']:.4f}, final_𝒢={v1_result['final_g']:.4f}, "
          f"viable_fraction={v1_result['viable_fraction']:.3f}")

    # Robustness
    print("\n  Robustness (10 seeds):")
    viable_fracs_v2 = []
    viable_fracs_v1 = []
    for s in range(10):
        c = V2Phase2Config(seed=s)
        r = run_phase2(c)
        viable_fracs_v2.append(r["viable_fraction"])
        r1 = run_v1_comparison(c)
        viable_fracs_v1.append(r1["viable_fraction"])
    print(f"    V2 viable: {np.mean(viable_fracs_v2):.3f} ± {np.std(viable_fracs_v2):.3f}")
    print(f"    V1 viable: {np.mean(viable_fracs_v1):.3f} ± {np.std(viable_fracs_v1):.3f}")

    # --------------------------------------------------------
    # Test 4: Necessity of V_𝒩
    # --------------------------------------------------------
    print("\n--- Test 4: With vs without V_𝒩 ---")

    cfg_noV = V2Phase2Config(enable_V=False)
    result_noV = run_phase2(cfg_noV)

    plot_phase2(result_noV, None,
                title="QMC V2 Phase 2 — Without V_𝒩 (diffusion only)",
                save_path="figures_v2/phase2_test4_no_V.png")

    print(f"  With V_𝒩:    viable={result['viable_fraction']:.3f}, "
          f"final_Var={result['final_mean_var']:.6f}")
    print(f"  Without V_𝒩: viable={result_noV['viable_fraction']:.3f}, "
          f"final_Var={result_noV['final_mean_var']:.6f}")

    # --------------------------------------------------------
    # Test 5: Distributional memory
    # --------------------------------------------------------
    print("\n--- Test 5: With vs without metric deformation ---")

    cfg_noMem = V2Phase2Config(enable_memory=False)
    result_noMem = run_phase2(cfg_noMem)

    print(f"  With memory:    viable={result['viable_fraction']:.3f}, "
          f"final_mean_R={result['final_mean_overlap']:.4f}")
    print(f"  Without memory: viable={result_noMem['viable_fraction']:.3f}, "
          f"final_mean_R={result_noMem['final_mean_overlap']:.4f}")

    # --------------------------------------------------------
    # Anti-fusion check
    # --------------------------------------------------------
    print("\n--- Anti-fusion check ---")
    print(f"  Final mean overlap R = {result['final_mean_overlap']:.4f}")
    print(f"  R in [0.2, 0.5]: {'YES' if 0.2 <= result['final_mean_overlap'] <= 0.5 else 'NO'}")
    print(f"  Max pairwise R = {max(h[-1] for h in result['pair_overlap_history'].values()):.4f}")
    print(f"  Min pairwise R = {min(h[-1] for h in result['pair_overlap_history'].values()):.4f}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"  Test 3 (Distributed vs V1 scalar):")
    print(f"    V2 viable fraction: {result['viable_fraction']:.3f}")
    print(f"    V1 viable fraction: {v1_result['viable_fraction']:.3f}")
    print(f"    Result: {'V2 > V1' if result['viable_fraction'] > v1_result['viable_fraction'] else 'INVESTIGATE'}")
    print(f"  Test 4 (Necessity of V_𝒩):")
    print(f"    With V_𝒩: {result['viable_fraction']:.3f}")
    print(f"    Without:  {result_noV['viable_fraction']:.3f}")
    print(f"    V_𝒩 necessary: {'YES' if result['viable_fraction'] > result_noV['viable_fraction'] else 'NO'}")
    print(f"  Test 5 (Distributional memory):")
    print(f"    With memory:    {result['viable_fraction']:.3f}")
    print(f"    Without memory: {result_noMem['viable_fraction']:.3f}")
    print(f"  Anti-fusion:")
    print(f"    R in viable corridor: {'YES' if 0.15 <= result['final_mean_overlap'] <= 0.6 else 'NO'}")
    print("=" * 70)
