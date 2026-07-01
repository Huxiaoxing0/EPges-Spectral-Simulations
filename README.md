# EPGES Spectral-Rank Simulations

This repository contains the reproducibility code and tabulated data for the
numerical checks accompanying the manuscript:

**Phase Adjacency, Spectral Rank, and Four-Dimensional Compatibility:
Bridge Standards for Dimensional Emergence**.

Repository: <https://github.com/Huxiaoxing0/EPges-Spectral-Simulations>

Archived release DOI: <https://doi.org/10.5281/zenodo.20745894>

The code supports the manuscript's finite-size heat-kernel checks, Bessel-flow
constants, anisotropic-weight tests, generator-rank comparisons, complete-graph
no-go diagnostic, and the nonseparable two-dimensional conductance stress test.

## Scope

The scripts are reproducibility aids for the spectral-rank calculations in the
paper. They do not derive a microscopic dynamics, a Lorentzian signature,
gravity, the Standard Model, or rank-four selection. Those questions are
identified in the manuscript as separate bridge problems.

## Repository Layout

```text
scripts/
  epges_spectral_validation.py
  epges_nonseparable_2d_bridge.py
  epges_random_local_bridge_benchmark.py
  epges_rank_sector_threshold_scan.py

outputs/
  figures_v12/
    fig7_bessel_dimension_flow.png
    fig1_core_proposition_v12.png
    fig4_anisotropy_corrected_v12.png
    fig5_generators_v12.png
    fig8_redundant_generators_v12.png
    fig6_complete_graph_v12.png
    epges_bessel_constants_v12.csv
```

The manuscript source package may use cleaned figure names, but the numerical
content is the same as the files generated here.

## Requirements

Python 3.10 or later is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `numpy`
- `matplotlib`
- `Pillow`

## Reproduce Manuscript Figures and Tables

### Figures 1-6 and Bessel constants

Run:

```bash
python scripts/epges_spectral_validation.py
```

This script uses exact finite heat-kernel sums and closed Bessel expressions.
It does not use Monte Carlo sampling.

Generated manuscript outputs include:

- `outputs/figures_v12/fig7_bessel_dimension_flow.png`
- `outputs/figures_v12/fig1_core_proposition_v12.png`
- `outputs/figures_v12/fig4_anisotropy_corrected_v12.png`
- `outputs/figures_v12/fig5_generators_v12.png`
- `outputs/figures_v12/fig8_redundant_generators_v12.png`
- `outputs/figures_v12/fig6_complete_graph_v12.png`
- `outputs/figures_v12/epges_bessel_constants_v12.csv`

Expected Bessel-flow constants:

```text
sigma_c = 0.851189972436
d_s^max / n = 1.217782494496
n=1: no d_s=2 crossing
n=2: sigma* = 0.394989210672
n=3: sigma* = 0.209702355585
n=4: sigma* = 0.146128850640
```

### Nonseparable two-dimensional conductance bridge check

Run:

```bash
python scripts/epges_nonseparable_2d_bridge.py
```

This script generates the nonseparable uniformly elliptic two-dimensional
stress test reported in the manuscript discussion, including the spectral
plateau data and homogenized conductance tables.

Expected output files include:

- `epges_nonsep2d_spectral.csv`
- `epges_nonsep2d_homogenized.csv`
- `fig9_nonseparable_2d_bridge.png`

Depending on the script version, these files may be written under an `outputs`
or `outputs/figures_v12` directory.

## Auxiliary Exploratory Diagnostics

The following scripts are included for transparency but are not used as proofs
or as numbered manuscript figures:

```bash
python scripts/epges_random_local_bridge_benchmark.py
python scripts/epges_rank_sector_threshold_scan.py
```

- `epges_random_local_bridge_benchmark.py` explores small finite random local
  bridge toy models using dense graph Laplacians.
- `epges_rank_sector_threshold_scan.py` explores a toy free-energy threshold
  scan for rank-sector selection.

These auxiliary diagnostics should be read as exploratory checks only. The
analytic claims of the paper rest on the heat-kernel formulae, bridge
conditions, and cited homogenization results stated in the manuscript.

## Manuscript-to-Code Map

| Manuscript item | Script |
| --- | --- |
| Bessel-flow constants and finite phase-torus checks | `scripts/epges_spectral_validation.py` |
| Anisotropic-weight and generator-rank finite checks | `scripts/epges_spectral_validation.py` |
| Complete-graph no-go diagnostic | `scripts/epges_spectral_validation.py` |
| Nonseparable 2D bridge stress test | `scripts/epges_nonseparable_2d_bridge.py` |
| Auxiliary random local bridge benchmark | `scripts/epges_random_local_bridge_benchmark.py` |
| Auxiliary rank-sector threshold scan | `scripts/epges_rank_sector_threshold_scan.py` |

## Citation

If using the archived code package, cite the Zenodo DOI:

<https://doi.org/10.5281/zenodo.20745894>
