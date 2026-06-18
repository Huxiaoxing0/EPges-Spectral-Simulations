# EPGES Spectral-Dimension Validation Code

This repository contains reproducibility code for the numerical and exact
spectral checks used in the EPGES manuscript:

**Euler-Phase Geometry of Emergent Spacetime: A Spectral-Rank
Consistency Criterion for Four-Dimensionality**.

Repository: https://github.com/Huxiaoxing0/EPges-Spectral-Simulations

Archived release DOI: https://doi.org/10.5281/zenodo.20621711

The code illustrates the intrinsic phase-torus proposition, finite-size
convergence, anisotropic weights, generator-set dependence, complete-graph
no-go behavior, random local conductance diagnostics, and a toy rank-sector
selection scan.

## Scripts

- `scripts/epges_spectral_validation.py`
  - Generates the main figures used in the manuscript.
  - Uses exact heat-kernel sums and closed Bessel formulae.
  - No Monte Carlo sampling is used.
  - Outputs figures and constants to `outputs/figures_v12/`.

- `scripts/epges_random_local_bridge_benchmark.py`
  - Explores a local random conductance bridge toy model.
  - Uses dense graph Laplacians and exact eigenvalues for small finite systems.
  - Outputs CSV files plus SVG/PNG diagnostics to `outputs/`.

- `scripts/epges_nonseparable_2d_bridge.py`
  - Numerical evidence for Conjecture 1 (random local phase-conductance bridge)
    in the nonseparable case: i.i.d. uniformly elliptic conductances on every
    edge of `(Z_N)^2`.
  - Compares two independent exact computations: the graph spectrum (quenched
    `d_s(sigma)` plateau converging to 2) and the periodic discrete corrector
    (cell) problem (homogenized matrix `A_hom`, rank 2). Verifies that
    `D_eff = sqrt(det A_hom)` tracks the geometric mean and is strictly inside
    the Wiener bounds -- i.e. `A_hom` is not the coordinatewise harmonic mean --
    and that the heat-trace plateau constant matches `(det A_hom)^{-1/2}`.
  - numpy + matplotlib only; no Monte Carlo dynamics (exact spectra and an exact
    CG corrector solve). Outputs `outputs/epges_nonsep2d_spectral.csv`,
    `outputs/epges_nonsep2d_homogenized.csv`, and
    `outputs/epges_nonsep2d_bridge.png`.

- `scripts/epges_rank_sector_threshold_scan.py`
  - Implements a toy free-energy threshold scan for rank-sector selection.
  - Outputs CSV tables and an SVG phase diagram to `outputs/`.

## Requirements

Python 3.10 or later is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

The main spectral script requires `numpy` and `matplotlib`.
The random local bridge script additionally requires `Pillow`.

## Reproduce the manuscript figures

From the repository root:

```bash
python scripts/epges_spectral_validation.py
```

Expected key constants:

```text
sigma_c = 0.851189972436
d_s^max / n = 1.217782494496
n=1: no d_s=2 crossing
n=2: sigma* = 0.394989210672
n=3: sigma* = 0.209702355585
n=4: sigma* = 0.146128850640
```

Generated files include:

- `outputs/figures_v12/fig7_bessel_dimension_flow.png`
- `outputs/figures_v12/fig1_core_proposition_v12.png`
- `outputs/figures_v12/fig4_anisotropy_corrected_v12.png`
- `outputs/figures_v12/fig5_generators_v12.png`
- `outputs/figures_v12/fig8_redundant_generators_v12.png`
- `outputs/figures_v12/fig6_complete_graph_v12.png`
- `outputs/figures_v12/epges_bessel_constants_v12.csv`

## Run auxiliary diagnostics

```bash
python scripts/epges_random_local_bridge_benchmark.py
python scripts/epges_rank_sector_threshold_scan.py
python scripts/epges_nonseparable_2d_bridge.py
```

These auxiliary scripts are exploratory. They are not used as proofs in the
paper; they illustrate finite-size and toy-model behavior. The nonseparable 2D
script in particular supplies numerical *evidence* for Conjecture 1, not a proof.

## Scientific scope

The code supports exact and finite-size checks for spectral-dimension claims.
It does **not** derive the Standard Model, gravity, the Born rule, or a complete
microscopic dynamics. The main analytic claims remain the heat-kernel formulae
and consistency filters stated in the manuscript.
