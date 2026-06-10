# EPGES Spectral-Dimension Validation Code

This repository contains reproducibility code for the numerical and exact
spectral checks used in the EPGES manuscript:

**Euler-Phase Geometry of Emergent Spacetime: A Spectral-Rank
Consistency Criterion for Four-Dimensionality**.

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
```

These auxiliary scripts are exploratory. They are not used as proofs in the
paper; they illustrate finite-size and toy-model behavior.

## Scientific scope

The code supports exact and finite-size checks for spectral-dimension claims.
It does **not** derive the Standard Model, gravity, the Born rule, or a complete
microscopic dynamics. The main analytic claims remain the heat-kernel formulae
and consistency filters stated in the manuscript.
