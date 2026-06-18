"""
Nonseparable 2D random-conductance bridge: numerical evidence for Conjecture 1.

The manuscript proves the spectral-rank/continuum bridge D = d_s = rank A_hom
only in controlled local abelian classes (regular tori, bounded directional
conductances, separable layered conductances, finite-generator abelian graphs,
and uniformly elliptic random finite-generator models). The genuinely
nonseparable random-conductance case is left as Conjecture 1.

This script supplies controlled numerical evidence for that conjecture in the
simplest nonseparable setting: independent identically distributed (i.i.d.),
uniformly elliptic conductances on every edge of a 2D phase torus (Z_N)^2. Two
independent computations are compared:

  (A) Spectral side. The exact graph spectrum gives the quenched return
      probability and spectral dimension d_s(sigma). We exhibit a stable
      d_s ~= 2 plateau and show that it sharpens (does not drift) as N grows,
      arguing the plateau is not a finite-size artifact -- the precise failure
      mode the conjecture warns about.

  (B) PDE-homogenization side. The deterministic homogenized matrix A_hom is
      computed independently, per realization, by solving the periodic discrete
      corrector (cell) problem. We verify:
        * rank A_hom = 2 (uniform ellipticity is preserved);
        * the heat-trace plateau constant matches (det A_hom)^{-1/2}, i.e. the
          spectral diffusion "sees" the homogenized elliptic form; and
        * D_eff = sqrt(det A_hom) lies strictly inside the Wiener bounds
          [harmonic mean, arithmetic mean] and tracks the geometric mean, NOT
          the coordinatewise harmonic mean. This is the nonseparable signature
          the manuscript notes is special to n = 1 / layered / separable cases.

Together (A) and (B) give a verified nonseparable instance of the bridge
beyond the separable class of Proposition 7, in numpy alone (no Monte Carlo
dynamics; exact spectra and an exact linear corrector solve).
"""

from __future__ import annotations

from pathlib import Path
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# ----------------------------------------------------------------------
# i.i.d. uniformly elliptic conductance field on the 2D torus.
# Normalized so the geometric mean is exactly 1; harmonic < 1 < arithmetic.
# ----------------------------------------------------------------------
def sample_conductances(N, s, seed, clip=3.0):
    """Return (cx, cy): horizontal/vertical edge conductances on (Z_N)^2.

    cx[i, j] is the conductance of the edge (i, j) -- (i+1, j).
    cy[i, j] is the conductance of the edge (i, j) -- (i, j+1).
    Lognormal exp(s Z), Z ~ N(0, 1), clipped to [exp(-clip s), exp(+clip s)]
    to enforce uniform ellipticity 0 < c1 <= c <= c2 < infinity.
    """
    rng = np.random.default_rng(seed)
    lo, hi = math.exp(-clip * s), math.exp(clip * s)
    cx = np.clip(np.exp(s * rng.standard_normal((N, N))), lo, hi)
    cy = np.clip(np.exp(s * rng.standard_normal((N, N))), lo, hi)
    return cx, cy


def conductance_means(s, clip=3.0, n=4_000_000, seed=0):
    """Wiener bounds and geometric mean of the clipped lognormal, by sampling."""
    rng = np.random.default_rng(seed)
    lo, hi = math.exp(-clip * s), math.exp(clip * s)
    c = np.clip(np.exp(s * rng.standard_normal(n)), lo, hi)
    return {
        "harmonic": float(1.0 / np.mean(1.0 / c)),
        "geometric": float(np.exp(np.mean(np.log(c)))),
        "arithmetic": float(np.mean(c)),
    }


# ----------------------------------------------------------------------
# Weighted graph Laplacian as a matvec on N x N grid functions (rolls only).
# (L u)(i,j) = cx[i,j] (u-u_E) + cx_W (u-u_W) + cy[i,j] (u-u_N) + cy_S (u-u_S)
# ----------------------------------------------------------------------
def laplacian_matvec(u, cx, cy):
    cx_W = np.roll(cx, 1, axis=0)
    cy_S = np.roll(cy, 1, axis=1)
    u_E = np.roll(u, -1, axis=0)
    u_W = np.roll(u, 1, axis=0)
    u_N = np.roll(u, -1, axis=1)
    u_S = np.roll(u, 1, axis=1)
    return cx * (u - u_E) + cx_W * (u - u_W) + cy * (u - u_N) + cy_S * (u - u_S)


def cg_solve(cx, cy, b, tol=1e-10, maxiter=20000):
    """Conjugate gradient for L u = b on the zero-mean subspace (L is SPSD with
    constant nullspace; b is already zero-mean)."""
    u = np.zeros_like(b)
    r = b - laplacian_matvec(u, cx, cy)
    r -= r.mean()
    p = r.copy()
    rs = float(np.vdot(r, r).real)
    b_norm = math.sqrt(float(np.vdot(b, b).real)) + 1e-300
    for _ in range(maxiter):
        Ap = laplacian_matvec(p, cx, cy)
        Ap -= Ap.mean()
        denom = float(np.vdot(p, Ap).real)
        if denom <= 0:
            break
        alpha = rs / denom
        u += alpha * p
        r -= alpha * Ap
        rs_new = float(np.vdot(r, r).real)
        if math.sqrt(rs_new) <= tol * b_norm:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    u -= u.mean()
    return u


def homogenized_matrix(cx, cy):
    """Exact periodic-corrector homogenized 2x2 matrix A_hom for one realization."""
    N = cx.shape[0]
    # Cell problem, direction e_1: L u1 = cx - cx_W.
    b1 = cx - np.roll(cx, 1, axis=0)
    u1 = cg_solve(cx, cy, b1)
    # Cell problem, direction e_2: L u2 = cy - cy_S.
    b2 = cy - np.roll(cy, 1, axis=1)
    u2 = cg_solve(cx, cy, b2)

    # Edge fields g = (imposed unit field along direction) + corrector gradient.
    def grads(u):
        return np.roll(u, -1, axis=0) - u, np.roll(u, -1, axis=1) - u  # (dx, dy)

    d1x, d1y = grads(u1)
    d2x, d2y = grads(u2)
    A = np.empty((2, 2))
    A[0, 0] = np.mean(cx * (1.0 + d1x))
    A[1, 0] = np.mean(cy * (0.0 + d1y))
    A[0, 1] = np.mean(cx * (0.0 + d2x))
    A[1, 1] = np.mean(cy * (1.0 + d2y))
    return 0.5 * (A + A.T)  # symmetrize (residual asymmetry ~ CG tolerance)


# ----------------------------------------------------------------------
# Spectral side: exact dense spectrum -> return probability -> d_s(sigma).
# ----------------------------------------------------------------------
def dense_laplacian(cx, cy):
    N = cx.shape[0]
    V = N * N
    L = np.zeros((V, V))

    def idx(i, j):
        return (i % N) * N + (j % N)

    for i in range(N):
        for j in range(N):
            a = idx(i, j)
            for (di, dj, c) in ((1, 0, cx[i, j]), (0, 1, cy[i, j])):
                b = idx(i + di, j + dj)
                L[a, a] += c
                L[b, b] += c
                L[a, b] -= c
                L[b, a] -= c
    return L


def spectral_dimension(evals, sigmas):
    P = np.array([np.mean(np.exp(-s * evals)) for s in sigmas])
    ds = -2.0 * np.gradient(np.log(P), np.log(sigmas))
    return P, ds


def plateau_point(sigmas, ds_mean, N):
    """Locate the genuine scaling plateau as the flattest point of d_s(sigma)
    in the band 3 <= sigma <= N^2/30, i.e. above the lattice-scale overshoot
    (sigma_c ~ 0.85) and below the finite-volume mixing time (~ N^2/40). The
    plateau is where |d d_s / d log sigma| is smallest; reporting this point
    avoids contaminating the estimate with the overshoot tail or the mixing
    downturn. Returns (sigma_star, ds_star, idx)."""
    lo, hi = 3.0, max(6.0, N * N / 30.0)
    band = np.where((sigmas >= lo) & (sigmas <= hi))[0]
    slope = np.abs(np.gradient(ds_mean, np.log(sigmas)))
    idx = band[np.argmin(slope[band])]
    return float(sigmas[idx]), float(ds_mean[idx]), int(idx)


def main():
    s = 0.8                       # lognormal disorder (geometric mean = 1)
    clip = 3.0
    means = conductance_means(s, clip)
    sigmas = np.logspace(math.log10(0.05), math.log10(60.0), 220)

    # (A) Spectral plateau and its finite-size sharpening.
    spectral_configs = [(16, 10), (24, 8), (32, 6), (40, 5)]
    spectral_curves = {}
    spectral_rows = []
    for N, n_seed in spectral_configs:
        ds_stack, P_stack, det_stack = [], [], []
        for seed in range(n_seed):
            cx, cy = sample_conductances(N, s, 1000 + seed, clip)
            evals = np.linalg.eigvalsh(dense_laplacian(cx, cy))  # computed once
            P, ds = spectral_dimension(evals, sigmas)
            ds_stack.append(ds)
            P_stack.append(P)
            det_stack.append(np.linalg.det(homogenized_matrix(cx, cy)))
        ds_mean = np.mean(ds_stack, axis=0)
        ds_sd = np.std(ds_stack, axis=0)
        P_mean = np.mean(P_stack, axis=0)
        spectral_curves[N] = (ds_mean, ds_sd)
        sigma_star, ds_plateau, idx = plateau_point(sigmas, ds_mean, N)
        ds_plateau_sd = float(ds_sd[idx])
        det_mean = float(np.mean(det_stack))
        # Heat-trace plateau constant vs (det A_hom)^{-1/2}: on a 2D plateau,
        # P(sigma) ~ (4 pi sigma)^{-1} (det A_hom)^{-1/2}. Evaluate at sigma_star.
        const_meas = float((4 * np.pi * sigmas * P_mean)[idx])
        spectral_rows.append({
            "N": N, "nodes": N * N, "seeds": n_seed,
            "sigma_star": sigma_star,
            "ds_plateau": ds_plateau, "ds_plateau_sd": ds_plateau_sd,
            "det_Ahom_mean": det_mean,
            "heat_const_measured": const_meas,
            "heat_const_predicted": float(det_mean ** -0.5),
        })
        print(f"[spectral] N={N:3d} nodes={N*N:5d} seeds={n_seed} "
              f"sigma*={sigma_star:6.2f} d_s={ds_plateau:.3f}+/-{ds_plateau_sd:.3f} "
              f"det(A_hom)={det_mean:.4f} "
              f"const_meas={const_meas:.4f} const_pred={det_mean**-0.5:.4f}")

    # (B) Homogenized D_eff vs Wiener bounds, with finite-size convergence.
    hom_configs = [(16, 200), (32, 80), (64, 30), (96, 14)]
    hom_rows = []
    for N, n_seed in hom_configs:
        Deff, l1, l2 = [], [], []
        for seed in range(n_seed):
            cx, cy = sample_conductances(N, s, 7000 + seed, clip)
            A = homogenized_matrix(cx, cy)
            ev = np.linalg.eigvalsh(A)
            Deff.append(math.sqrt(max(np.linalg.det(A), 0.0)))
            l1.append(ev[0]); l2.append(ev[1])
        hom_rows.append({
            "N": N, "seeds": n_seed,
            "Deff_mean": float(np.mean(Deff)), "Deff_sd": float(np.std(Deff)),
            "lambda_min_mean": float(np.mean(l1)),
            "lambda_max_mean": float(np.mean(l2)),
            "harmonic": means["harmonic"], "geometric": means["geometric"],
            "arithmetic": means["arithmetic"],
        })
        print(f"[homog]    N={N:3d} seeds={n_seed:3d} "
              f"D_eff={np.mean(Deff):.4f}+/-{np.std(Deff):.4f} "
              f"(harm={means['harmonic']:.4f} geo={means['geometric']:.4f} "
              f"arith={means['arithmetic']:.4f})")

    # ---- write CSVs ----
    with (OUT / "epges_nonsep2d_spectral.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(spectral_rows[0].keys()))
        w.writeheader(); w.writerows(spectral_rows)
    with (OUT / "epges_nonsep2d_homogenized.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hom_rows[0].keys()))
        w.writeheader(); w.writerows(hom_rows)

    # ---- figure ----
    fig, (axD, axH) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for i, (N, (ds_mean, ds_sd)) in enumerate(spectral_curves.items()):
        axD.semilogx(sigmas, ds_mean, color=COL[i], label=fr"$N={N}$ ($N^2={N*N}$)")
        axD.fill_between(sigmas, ds_mean - ds_sd, ds_mean + ds_sd,
                         color=COL[i], alpha=0.12)
        sigma_star, ds_star, _ = plateau_point(sigmas, ds_mean, N)
        axD.plot(sigma_star, ds_star, "o", color=COL[i], ms=5)
    axD.axhline(2.0, color="gray", ls="--", alpha=0.7, label=r"$d_s=2$")
    axD.set_xlabel(r"diffusion time $\sigma$")
    axD.set_ylabel(r"$d_s(\sigma)$ (quenched)")
    axD.set_title(r"Nonseparable i.i.d. random conductances on $(\mathbb{Z}_N)^2$")
    axD.set_ylim(0, 3.0)
    axD.legend(fontsize=8, loc="lower left")

    Ns = [r["N"] for r in hom_rows]
    Deff = [r["Deff_mean"] for r in hom_rows]
    Dsd = [r["Deff_sd"] for r in hom_rows]
    axH.errorbar(Ns, Deff, yerr=Dsd, marker="o", color=COL[3], capsize=3,
                 label=r"$D_{\rm eff}=\sqrt{\det A_{\rm hom}}$ (corrector)")
    axH.axhline(means["harmonic"], color=COL[0], ls="--",
                label=fr"harmonic mean $={means['harmonic']:.3f}$")
    axH.axhline(means["geometric"], color=COL[2], ls=":",
                label=fr"geometric mean $={means['geometric']:.3f}$")
    axH.axhline(means["arithmetic"], color=COL[1], ls="--",
                label=fr"arithmetic mean $={means['arithmetic']:.3f}$")
    axH.set_xlabel(r"linear size $N$")
    axH.set_ylabel(r"effective diffusion constant")
    axH.set_title(r"$A_{\rm hom}$ is not the harmonic mean (corrector effect)")
    axH.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "epges_nonsep2d_bridge.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "epges_nonsep2d_bridge.png")


if __name__ == "__main__":
    main()
