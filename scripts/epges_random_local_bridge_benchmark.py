import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def torus_laplacian(rank, N, disorder=0.0, seed=0, return_weight_stats=False):
    """Dense nearest-neighbor weighted Laplacian on (Z_N)^rank."""
    rng = np.random.default_rng(seed)
    shape = (N,) * rank
    volume = N**rank
    L = np.zeros((volume, volume), dtype=float)
    weights = []

    def idx(coord):
        return np.ravel_multi_index(tuple(coord), shape)

    for coord_tuple in np.ndindex(shape):
        coord = np.array(coord_tuple, dtype=int)
        i = idx(coord)
        for mu in range(rank):
            nbr = coord.copy()
            nbr[mu] = (nbr[mu] + 1) % N
            j = idx(nbr)
            if disorder > 0:
                w = math.exp(disorder * rng.normal() - 0.5 * disorder**2)
            else:
                w = 1.0
            weights.append(w)
            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w
    if return_weight_stats:
        arr = np.array(weights, dtype=float)
        return L, {
            "w_min": float(np.min(arr)),
            "w_max": float(np.max(arr)),
            "w_mean": float(np.mean(arr)),
        }
    return L


def complete_graph_spectrum(M):
    return np.array([0.0] + [float(M)] * (M - 1))


def product_torus_spectrum(rank, N):
    """Exact product-spectrum eigenvalues for a clean rank-r torus."""
    cycle = np.array([2.0 * (1.0 - math.cos(2.0 * math.pi * m / N)) for m in range(N)])
    vals = cycle.copy()
    for _ in range(rank - 1):
        vals = (vals[:, None] + cycle[None, :]).reshape(-1)
    return vals


def spectral_dimension(evals, sigmas):
    evals = np.asarray(evals)
    sigmas = np.asarray(sigmas)
    P = np.array([np.mean(np.exp(-s * evals)) for s in sigmas])
    logP = np.log(P)
    logs = np.log(sigmas)
    ds = -2.0 * np.gradient(logP, logs)
    return P, ds


def finite_plateau_score(sigmas, ds, target, lo=0.35, hi=2.5):
    mask = (sigmas >= lo) & (sigmas <= hi)
    if not np.any(mask):
        return float("nan"), float("nan")
    return float(np.mean(ds[mask] - target)), float(np.sqrt(np.mean((ds[mask] - target) ** 2)))


def polyline(points, xscale, yscale, xmin, ymin, height, color, width=1.8):
    coords = []
    for x, y in points:
        px = 70 + (math.log10(x) - xmin) * xscale
        py = 20 + height - (y - ymin) * yscale
        coords.append(f"{px:.2f},{py:.2f}")
    return f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="{width}" />'


def make_svg(path, sigmas, curves):
    width, height = 760, 470
    plot_w, plot_h = 620, 360
    xmin = math.log10(float(sigmas[0]))
    xmax = math.log10(float(sigmas[-1]))
    ymin, ymax = -0.2, 4.8
    xscale = plot_w / (xmax - xmin)
    yscale = plot_h / (ymax - ymin)

    colors = {
        "rank 2 clean": "#1f77b4",
        "rank 2 random": "#6baed6",
        "rank 3 clean": "#2ca02c",
        "rank 3 random": "#74c476",
        "rank 4 clean": "#d62728",
        "rank 4 random": "#fb6a4a",
        "complete graph": "#444444",
    }

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="70" y="20" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#222" stroke-width="1"/>',
    ]

    for y in [0, 1, 2, 3, 4]:
        py = 20 + plot_h - (y - ymin) * yscale
        lines.append(f'<line x1="70" y1="{py:.2f}" x2="{70+plot_w}" y2="{py:.2f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="48" y="{py+4:.2f}" font-size="12" text-anchor="end">{y}</text>')

    for x in [0.05, 0.1, 0.3, 1, 3, 10, 20]:
        if sigmas[0] <= x <= sigmas[-1]:
            px = 70 + (math.log10(x) - xmin) * xscale
            lines.append(f'<line x1="{px:.2f}" y1="20" x2="{px:.2f}" y2="{20+plot_h}" stroke="#eee" stroke-width="1"/>')
            lines.append(f'<text x="{px:.2f}" y="{20+plot_h+20}" font-size="11" text-anchor="middle">{x:g}</text>')

    for label, ds in curves.items():
        pts = list(zip(sigmas, ds))
        lines.append(polyline(pts, xscale, yscale, xmin, ymin, plot_h, colors[label], 2.1 if "random" in label else 1.5))

    lines.append('<text x="380" y="448" font-size="14" text-anchor="middle">diffusion time sigma</text>')
    lines.append('<text x="20" y="205" font-size="14" text-anchor="middle" transform="rotate(-90 20 205)">spectral dimension d_s(sigma)</text>')
    lines.append('<text x="70" y="15" font-size="13">Random local conductance preserves the rank plateau; nonlocal complete graph does not.</text>')

    legend_x, legend_y = 505, 38
    for k, label in enumerate(curves):
        y = legend_y + 20 * k
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+26}" y2="{y}" stroke="{colors[label]}" stroke-width="2.2"/>')
        lines.append(f'<text x="{legend_x+34}" y="{y+4}" font-size="12">{label}</text>')

    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def make_png(path, sigmas, curves):
    width, height = 1520, 940
    margin_l, margin_t = 140, 50
    plot_w, plot_h = 1240, 720
    xmin = math.log10(float(sigmas[0]))
    xmax = math.log10(float(sigmas[-1]))
    ymin, ymax = -0.2, 4.8
    xscale = plot_w / (xmax - xmin)
    yscale = plot_h / (ymax - ymin)
    colors = {
        "rank 2 clean": (31, 119, 180),
        "rank 2 random": (107, 174, 214),
        "rank 3 clean": (44, 160, 44),
        "rank 3 random": (116, 196, 118),
        "rank 4 clean": (214, 39, 40),
        "rank 4 random": (251, 106, 74),
        "complete graph": (68, 68, 68),
    }
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small = ImageFont.truetype("arial.ttf", 20)
        title = ImageFont.truetype("arial.ttf", 23)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
        title = ImageFont.load_default()

    draw.rectangle([margin_l, margin_t, margin_l + plot_w, margin_t + plot_h], fill=(250, 250, 250), outline=(30, 30, 30), width=2)
    for y in [0, 1, 2, 3, 4]:
        py = margin_t + plot_h - (y - ymin) * yscale
        draw.line([margin_l, py, margin_l + plot_w, py], fill=(220, 220, 220), width=2)
        draw.text((margin_l - 40, py - 12), str(y), fill=(20, 20, 20), font=small, anchor="ra")
    for x in [0.05, 0.1, 0.3, 1, 3, 10, 20]:
        if sigmas[0] <= x <= sigmas[-1]:
            px = margin_l + (math.log10(x) - xmin) * xscale
            draw.line([px, margin_t, px, margin_t + plot_h], fill=(235, 235, 235), width=2)
            draw.text((px, margin_t + plot_h + 34), f"{x:g}", fill=(20, 20, 20), font=small, anchor="mm")

    def xy(x, y):
        return (
            margin_l + (math.log10(float(x)) - xmin) * xscale,
            margin_t + plot_h - (float(y) - ymin) * yscale,
        )

    for label, ds in curves.items():
        pts = [xy(x, y) for x, y in zip(sigmas, ds)]
        draw.line(pts, fill=colors[label], width=5 if "random" in label else 3, joint="curve")

    draw.text((width / 2, height - 40), "diffusion time sigma", fill=(20, 20, 20), font=font, anchor="mm")
    draw.text((38, margin_t + plot_h / 2), "spectral dimension d_s(sigma)", fill=(20, 20, 20), font=font, anchor="mm")
    draw.text((margin_l, 22), "Random local conductance preserves the rank trend; nonlocal complete graph does not.", fill=(20, 20, 20), font=title)

    legend_x, legend_y = 1010, 88
    for k, label in enumerate(curves):
        y = legend_y + 40 * k
        draw.line([legend_x, y, legend_x + 52, y], fill=colors[label], width=5)
        draw.text((legend_x + 68, y), label, fill=(20, 20, 20), font=small, anchor="lm")
    img.save(path)


def main():
    OUT.mkdir(exist_ok=True)
    sigmas = np.logspace(math.log10(0.035), math.log10(20.0), 160)
    configs = [
        ("rank 2 clean", 2, 48, 0.0, [0]),
        ("rank 2 random", 2, 24, 0.7, [11, 12, 13]),
        ("rank 3 clean", 3, 18, 0.0, [0]),
        ("rank 3 random", 3, 12, 0.7, [21, 22, 23]),
        ("rank 4 clean", 4, 10, 0.0, [0]),
        ("rank 4 random", 4, 7, 0.7, [31]),
    ]

    curves = {}
    rows = []
    for label, rank, N, disorder, seeds in configs:
        all_ds = []
        all_P = []
        w_min_values = []
        w_max_values = []
        for seed in seeds:
            if disorder == 0:
                evals = product_torus_spectrum(rank, N)
                w_stats = {"w_min": 1.0, "w_max": 1.0}
            else:
                L, w_stats = torus_laplacian(rank, N, disorder, seed, return_weight_stats=True)
                evals = np.linalg.eigvalsh(L)
            w_min_values.append(w_stats["w_min"])
            w_max_values.append(w_stats["w_max"])
            P, ds = spectral_dimension(evals, sigmas)
            all_ds.append(ds)
            all_P.append(P)
        ds_mean = np.mean(all_ds, axis=0)
        P_mean = np.mean(all_P, axis=0)
        curves[label] = ds_mean
        # For a full-rank finite-size plateau all active directions must remain
        # unsaturated. Fast directions saturate first, so the conservative upper
        # window is controlled by w_max, while w_min controls the late slow tail.
        mean_wmax = float(np.mean(w_max_values))
        plateau_hi = min(2.5, N**2 / (6.0 * mean_wmax))
        mean_err, rms_err = finite_plateau_score(sigmas, ds_mean, rank, hi=plateau_hi)
        rows.append({
            "model": label,
            "rank_target": rank,
            "N": N,
            "volume": N**rank,
            "disorder_lognormal_sigma": disorder,
            "samples": len(seeds),
            "w_min_mean": float(np.mean(w_min_values)),
            "w_max_mean": mean_wmax,
            "plateau_window_lo": 0.35,
            "plateau_window_hi": plateau_hi,
            "plateau_mean_error": mean_err,
            "plateau_rms_error": rms_err,
            "ds_sigma_0p3": float(np.interp(0.3, sigmas, ds_mean)),
            "ds_sigma_1": float(np.interp(1.0, sigmas, ds_mean)),
            "ds_sigma_3": float(np.interp(3.0, sigmas, ds_mean)),
        })

    evals_complete = complete_graph_spectrum(600)
    _, ds_complete = spectral_dimension(evals_complete, sigmas)
    curves["complete graph"] = ds_complete
    rows.append({
        "model": "complete graph",
        "rank_target": 0,
        "N": 600,
        "volume": 600,
        "disorder_lognormal_sigma": 0.0,
        "samples": 1,
        "w_min_mean": "",
        "w_max_mean": "",
        "plateau_window_lo": 0.35,
        "plateau_window_hi": 2.5,
        "plateau_mean_error": float(np.mean(ds_complete[(sigmas >= 0.35) & (sigmas <= 2.5)])),
        "plateau_rms_error": float(np.sqrt(np.mean(ds_complete[(sigmas >= 0.35) & (sigmas <= 2.5)] ** 2))),
        "ds_sigma_0p3": float(np.interp(0.3, sigmas, ds_complete)),
        "ds_sigma_1": float(np.interp(1.0, sigmas, ds_complete)),
        "ds_sigma_3": float(np.interp(3.0, sigmas, ds_complete)),
    })

    csv_path = OUT / "epges_random_local_bridge_values.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    curve_csv = OUT / "epges_random_local_bridge_curves.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sigma"] + list(curves.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, s in enumerate(sigmas):
            row = {"sigma": float(s)}
            for label, ds in curves.items():
                row[label] = float(ds[i])
            writer.writerow(row)

    svg_path = OUT / "epges_random_local_bridge_ds.svg"
    make_svg(svg_path, sigmas, curves)
    png_path = OUT / "epges_random_local_bridge_ds.png"
    make_png(png_path, sigmas, curves)
    print(csv_path)
    print(curve_csv)
    print(svg_path)
    print(png_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
