"""
EPGES spectral-dimension validation suite, v12.

This script accompanies the FOP submission draft.  It illustrates the
closed phase-torus proposition, finite-size convergence, anisotropic
weights, generator-set dependence, the complete-graph no-go result, and
the exact infinite-volume Bessel dimension flow.

Numerics are exact heat-kernel sums using the product eigenvalue
structure; no Monte Carlo sampling is used.
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
OUT = ROOT / "outputs" / "figures_v12"
OUT.mkdir(parents=True, exist_ok=True)

NUM_SIGMA = 320
SIGMA_LO = 3.0
SIGMA_DENOM = 6.0
PLATEAU_MIN_PTS = 12

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "lines.linewidth": 1.7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def i0_series(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = x * x / 4.0
    term = np.ones_like(x)
    total = np.ones_like(x)
    for k in range(1, 200):
        term = term * y / (k * k)
        total_new = total + term
        if np.all(np.abs(term) < 1e-15 * np.maximum(1.0, np.abs(total_new))):
            return total_new
        total = total_new
    return total


def i1_series(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = x * x / 4.0
    term = x / 2.0
    total = term.copy()
    for k in range(1, 200):
        term = term * y / (k * (k + 1))
        total_new = total + term
        if np.all(np.abs(term) < 1e-15 * np.maximum(1.0, np.abs(total_new))):
            return total_new
        total = total_new
    return total


def W(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    small = x < 30.0
    out[small] = i1_series(x[small]) / i0_series(x[small])
    xs = x[~small]
    if xs.size:
        # Large-x asymptotic for I1/I0:
        # 1 - 1/(2x) - 1/(8x^2) - 1/(8x^3) - 25/(128x^4) + O(x^-5).
        out[~small] = (
            1.0
            - 1.0/(2.0*xs)
            - 1.0/(8.0*xs**2)
            - 1.0/(8.0*xs**3)
            - 25.0/(128.0*xs**4)
        )
    return out


def zn_eigenvalues(N: int) -> np.ndarray:
    m = np.arange(N, dtype=float)
    return 2.0 * (1.0 - np.cos(2.0 * np.pi * m / N))


def torus_P(sigmas: np.ndarray, N: int, n: int, weights=None) -> np.ndarray:
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=float)
    lam = zn_eigenvalues(N)
    logP = np.zeros_like(sigmas, dtype=float)
    for wj in weights:
        E = np.exp(-np.outer(sigmas, wj * lam))
        logP += np.log(np.maximum(np.mean(E, axis=1), 1e-300))
    return np.exp(logP)


def cycle_P(sigmas: np.ndarray, M: int) -> np.ndarray:
    lam = zn_eigenvalues(M)
    return np.mean(np.exp(-np.outer(sigmas, lam)), axis=1)


def complete_P(sigmas: np.ndarray, N: int) -> np.ndarray:
    return (1.0 + (N - 1.0) * np.exp(-N * sigmas)) / N


def complete_ds(sigmas: np.ndarray, N: int) -> np.ndarray:
    return 2.0 * N * sigmas * (N - 1.0) / (np.exp(N * sigmas) + N - 1.0)


def finite_generator_P(sigmas: np.ndarray, N: int, generators, weights=None) -> np.ndarray:
    """Return probability for a translation-invariant Cayley graph on (Z_N)^2.

    ``generators`` is a list of unoriented generator pairs represented by
    integer vectors.  Each pair contributes edges in both +/- directions, so
    the eigenvalue contribution is 2 w [1 - cos(k . s)].
    """
    gens = np.asarray(generators, dtype=int)
    if weights is None:
        weights = np.ones(len(gens), dtype=float)
    weights = np.asarray(weights, dtype=float)
    m1 = np.arange(N, dtype=float)
    m2 = np.arange(N, dtype=float)
    k1, k2 = np.meshgrid(m1, m2, indexing="ij")
    lam = np.zeros((N, N), dtype=float)
    for (s1, s2), w in zip(gens, weights):
        phase = 2.0 * np.pi * (s1 * k1 + s2 * k2) / N
        lam += 2.0 * w * (1.0 - np.cos(phase))
    return np.mean(np.exp(-np.outer(sigmas, lam.ravel())), axis=1)


def generator_rank(generators) -> int:
    return int(np.linalg.matrix_rank(np.asarray(generators, dtype=float)))


def ds_numerical(sigmas: np.ndarray, P: np.ndarray) -> np.ndarray:
    return -2.0 * np.gradient(np.log(np.maximum(P, 1e-300)), np.log(sigmas))


def ds_bessel(sigmas: np.ndarray, n: int) -> np.ndarray:
    return 4.0 * n * sigmas * (1.0 - W(2.0 * sigmas))


def ds_weighted_bessel(sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = np.zeros_like(sigmas, dtype=float)
    for wj in weights:
        total += wj * (1.0 - W(2.0 * wj * sigmas))
    return 4.0 * sigmas * total


def find_plateau(sigmas, P, d_s, N_eff, sigma_upper=None):
    hi = N_eff**2 / SIGMA_DENOM if sigma_upper is None else sigma_upper
    mask = (sigmas > SIGMA_LO) & (sigmas < hi) & np.isfinite(d_s) & (d_s > 0)
    if np.sum(mask) < PLATEAU_MIN_PTS:
        return np.nan, np.nan, mask
    return float(np.median(d_s[mask])), float(np.std(d_s[mask])), mask


def bisect(f, a, b, eps=1e-12):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Root is not bracketed.")
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < eps or abs(b - a) < eps:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def overshoot_constants():
    sigma_c = bisect(
        lambda s: 2.0 * s * (1.0 - float(W(2.0 * s))**2) - 1.0,
        0.5,
        2.0,
    )
    ratio = float(ds_bessel(np.array([sigma_c]), 1)[0])
    return sigma_c, ratio


def constants_report():
    sigma_c, ratio = overshoot_constants()
    print(f"sigma_c = {sigma_c:.12f}")
    print(f"d_s^max / n = {ratio:.12f}")

    rows = [{
        "quantity": "universal_overshoot_sigma_c",
        "n": "",
        "value": sigma_c,
        "description": "Root of 2 sigma [1 - (I1(2 sigma)/I0(2 sigma))^2] = 1.",
    }, {
        "quantity": "universal_overshoot_ratio",
        "n": "",
        "value": ratio,
        "description": "Maximum value of d_s(sigma)/n on the infinite phase torus.",
    }]

    for n in [1, 2, 3, 4]:
        dmax = ratio * n
        if dmax > 2:
            root = bisect(lambda s, n=n: float(ds_bessel(np.array([s]), n)[0]) - 2.0, 1e-10, sigma_c)
            print(f"n={n}: dmax={dmax:.12f}, d_s=2 crossing sigma*={root:.12f}")
            rows.append({
                "quantity": "ds_equals_2_crossing",
                "n": n,
                "value": root,
                "description": "Finite-scale crossing of the dimension-flow curve through d_s=2; diagnostic only, not a Polya recurrence theorem.",
            })
        else:
            print(f"n={n}: dmax={dmax:.12f}, no d_s=2 crossing")
            rows.append({
                "quantity": "ds_equals_2_crossing",
                "n": n,
                "value": "",
                "description": "No crossing because the universal overshoot remains below d_s=2.",
            })

    with (OUT / "epges_bessel_constants_v12.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["quantity", "n", "value", "description"])
        writer.writeheader()
        writer.writerows(rows)


def plot_bessel_flow():
    sigmas = np.logspace(-3, 2, 800)
    sigma_c, ratio = overshoot_constants()

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for i, n in enumerate([1, 2, 3, 4]):
        d = ds_bessel(sigmas, n)
        ax.semilogx(sigmas, d, color=COL[i], label=fr"$n={n}$")
        ax.axhline(n, color=COL[i], ls=":", alpha=0.35)
        ax.plot(sigma_c, ratio * n, "o", color=COL[i], ms=4)
        if ratio * n > 2:
            root = bisect(lambda s, n=n: float(ds_bessel(np.array([s]), n)[0]) - 2.0, 1e-10, sigma_c)
            ax.plot(root, 2.0, marker="x", color=COL[i], ms=5)
    ax.axvline(sigma_c, color="k", ls="--", alpha=0.45,
               label=fr"$\sigma_c={sigma_c:.3f}$")
    ax.axhline(2, color="gray", ls="--", alpha=0.35)
    ax.set_xlabel(r"Diffusion time $\sigma$")
    ax.set_ylabel(r"$d_s^{(\infty)}(\sigma)$")
    ax.set_title(r"Exact Bessel dimension flow on $\mathbb{Z}^n$")
    ax.set_ylim(0, 5.6)
    ax.legend(loc="best")
    ax.text(0.03, 0.93, fr"$d_s^{{\max}}/n={ratio:.3f}$",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(OUT / "fig7_bessel_dimension_flow.png", bbox_inches="tight")
    plt.close(fig)


def plot_core_proposition():
    cfgs = [(1, 100), (2, 50), (3, 30), (4, 20)]
    fig, (axP, axD) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for i, (n, N) in enumerate(cfgs):
        sigmas = np.logspace(0, np.log10(N**2 / SIGMA_DENOM), NUM_SIGMA)
        P = torus_P(sigmas, N, n)
        d_s = ds_numerical(sigmas, P)
        dE, dStd, _ = find_plateau(sigmas, P, d_s, N)
        axP.loglog(sigmas, P, color=COL[i], label=fr"$n={n},\,N={N}$")
        axP.loglog(sigmas, (4*np.pi*sigmas)**(-n/2), color=COL[i], lw=5, alpha=0.13)
        axD.semilogx(sigmas, d_s, color=COL[i],
                     label=fr"$n={n}: d_{{\rm eff}}={dE:.2f}\pm{dStd:.2f}$")
        axD.axhline(n, color=COL[i], ls=":", alpha=0.35)
    axP.set_xlabel(r"$\sigma$")
    axP.set_ylabel(r"$P(\sigma)$")
    axP.set_title(r"Return probability and Gaussian scaling")
    axP.legend(loc="upper right")
    axD.set_xlabel(r"$\sigma$")
    axD.set_ylabel(r"$d_s(\sigma)$")
    axD.set_title(r"Finite-torus spectral dimension")
    axD.set_ylim(0, 5.8)
    axD.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_core_proposition_v12.png", bbox_inches="tight")
    plt.close(fig)


def plot_anisotropy_corrected():
    N, n = 60, 4
    configs = [
        (np.ones(4), "isotropic (1,1,1,1)"),
        (np.array([1., 2., 3., 4.]), "moderate (1,2,3,4)"),
        (np.array([1., 5., 10., 20.]), "strong (1,5,10,20)"),
        (np.array([2., 2., 2., 2.]), "uniform (2,2,2,2)"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for i, (w, label) in enumerate(configs):
        # Fast directions saturate first; use w_max for the finite-size upper window.
        sig_hi = max(6.0, N**2 / (SIGMA_DENOM * float(np.max(w))))
        sigmas = np.logspace(0, np.log10(sig_hi), NUM_SIGMA)
        P = torus_P(sigmas, N, n, weights=w)
        d_s = ds_numerical(sigmas, P)
        ax.semilogx(sigmas, d_s, color=COL[i], label=label)
        ax.semilogx(sigmas, ds_weighted_bessel(sigmas, w),
                    color=COL[i], ls="--", alpha=0.55)
    ax.axhline(4, color="gray", ls="--", alpha=0.6, label=r"$d_s=4$")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$d_s(\sigma)$")
    ax.set_title(r"Positive anisotropic weights: finite-size corrected window")
    ax.set_ylim(0, 5.5)
    ax.text(0.03, 0.05, "solid: finite torus; dashed: infinite-volume Bessel flow",
            transform=ax.transAxes, ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
            fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_anisotropy_corrected_v12.png", bbox_inches="tight")
    plt.close(fig)


def plot_generators_and_complete():
    # Generator set.
    N, n = 10, 4
    M = N**n
    # The one-dimensional cycle has a much wider finite-size window than
    # (Z_N)^4 with the same node count, so the range is extended to show its
    # d_s ~= 1 plateau rather than only its early-time approach.
    sigmas = np.logspace(0, 5.2, NUM_SIGMA)
    P_torus = torus_P(sigmas, N, n)
    P_cycle = cycle_P(sigmas, M)
    ds_torus = ds_numerical(sigmas, P_torus)
    ds_cycle = ds_numerical(sigmas, P_cycle)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax1.loglog(sigmas, P_torus, color=COL[3], label=fr"$(\mathbb{{Z}}_{N})^4$")
    ax1.loglog(sigmas, P_cycle, color=COL[0], label=fr"$\mathbb{{Z}}_{{{M}}}$")
    ax1.set_xlabel(r"$\sigma$")
    ax1.set_ylabel(r"$P(\sigma)$")
    ax1.set_title("Same node count, different local generators")
    ax1.legend()
    ax2.semilogx(sigmas, ds_torus, color=COL[3], label=fr"$(\mathbb{{Z}}_{N})^4$")
    ax2.semilogx(sigmas, ds_cycle, color=COL[0], label=fr"$\mathbb{{Z}}_{{{M}}}$")
    ax2.axhline(4, color=COL[3], ls=":", alpha=0.4)
    ax2.axhline(1, color=COL[0], ls=":", alpha=0.4)
    ax2.set_xlabel(r"$\sigma$")
    ax2.set_ylabel(r"$d_s(\sigma)$")
    ax2.set_title("Generator set controls dimension")
    ax2.set_ylim(0, 5.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig5_generators_v12.png", bbox_inches="tight")
    plt.close(fig)


def plot_redundant_generators():
    # Use an odd period so the diagonal pair (1,1), (1,-1) is connected
    # modulo N; for even N it would split into parity sectors.
    N = 101
    sigmas = np.logspace(0, np.log10(N**2 / SIGMA_DENOM), NUM_SIGMA)
    configs = [
        ([(1, 0), (0, 1)], r"$S_1=\{\pm e_1,\pm e_2\}$", COL[0]),
        ([(1, 0), (0, 1), (1, 1)], r"$S_2=S_1\cup\{\pm(e_1+e_2)\}$", COL[1]),
        ([(1, 1), (1, -1)], r"$S_3=\{\pm(e_1+e_2),\pm(e_1-e_2)\}$", COL[2]),
    ]
    fig, (axP, axD) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for gens, label, color in configs:
        P = finite_generator_P(sigmas, N, gens)
        d_s = ds_numerical(sigmas, P)
        rank = generator_rank(gens)
        oriented_count = 2 * len(gens)
        axP.loglog(sigmas, P, color=color,
                   label=fr"{label}: $|S|={oriented_count},\,\mathrm{{rank}}={rank}$")
        axD.semilogx(sigmas, d_s, color=color,
                     label=fr"$|S|={oriented_count},\,\mathrm{{rank}}={rank}$")
    axP.loglog(sigmas, (4*np.pi*sigmas)**(-1), color="gray", ls="--",
               alpha=0.55, label=r"reference $\sigma^{-1}$")
    axP.set_xlabel(r"$\sigma$")
    axP.set_ylabel(r"$P(\sigma)$")
    axP.set_title(r"Redundant generators alter the diffusion tensor")
    axP.legend(fontsize=7)
    axD.axhline(2, color="gray", ls="--", alpha=0.6, label=r"$d_s=2$")
    axD.set_xlabel(r"$\sigma$")
    axD.set_ylabel(r"$d_s(\sigma)$")
    axD.set_title(r"Rank, not raw generator count, fixes $d_s$")
    axD.set_ylim(0, 3.0)
    axD.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_redundant_generators_v12.png", bbox_inches="tight")
    plt.close(fig)


def plot_complete_graph():
    # Complete graph no-go.
    Ns = [50, 100, 200, 500]
    fig, (axP, axD) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for i, NK in enumerate(Ns):
        sig = np.logspace(-4, np.log10(10.0/NK), 500)
        x = sig * NK
        axP.loglog(x, complete_P(sig, NK), color=COL[i], label=fr"$N={NK}$")
        axD.plot(x, complete_ds(sig, NK), color=COL[i], label=fr"$N={NK}$")
    axP.set_xlabel(r"Rescaled time $\sigma N$")
    axP.set_ylabel(r"$P_K(\sigma)$")
    axP.set_title("Complete graph: rapid crossover")
    axP.legend()
    axD.set_xlabel(r"Rescaled time $\sigma N$")
    axD.set_ylabel(r"$d_s(\sigma)$")
    axD.set_title("No extended finite-dimensional plateau")
    axD.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig6_complete_graph_v12.png", bbox_inches="tight")
    plt.close(fig)


def main():
    print("EPGES spectral validation v12")
    constants_report()
    plot_bessel_flow()
    plot_core_proposition()
    plot_anisotropy_corrected()
    plot_generators_and_complete()
    plot_redundant_generators()
    plot_complete_graph()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
