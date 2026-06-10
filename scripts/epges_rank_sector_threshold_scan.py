import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def compatibility_cost(r: int, p: float = 1.0) -> float:
    return abs(r - 4) ** p


def epsilon(r: int, eta: float) -> float:
    # Positive eta penalizes ranks above four and favors ranks below four.
    # Negative eta favors ranks above four.
    return eta * (r - 4)


def lambda_threshold(M: int, T: float, eta: float, p: float = 1.0) -> tuple[float, int, float]:
    if M < 4:
        raise ValueError("The rank-four sector is unavailable when M < 4.")
    best = 0.0
    best_r = 4
    best_num = 0.0
    s4 = log_comb(M, 4)
    e4 = epsilon(4, eta)
    for r in range(1, M + 1):
        if r == 4:
            continue
        C = compatibility_cost(r, p)
        if C == 0:
            continue
        numerator = T * (log_comb(M, r) - s4) - (epsilon(r, eta) - e4)
        bound = numerator / C
        if not math.isfinite(bound):
            continue
        if bound > best:
            best = bound
            best_r = r
            best_num = numerator
    return best, best_r, best_num


def selected_rank(M: int, T: float, eta: float, lam: float, p: float = 1.0) -> int:
    # For M < 4 the rank-four candidate does not exist; the function still
    # returns the least-cost available rank, but such cases are not used in the
    # threshold scan below.
    values = []
    for r in range(1, M + 1):
        f = epsilon(r, eta) - T * log_comb(M, r) + lam * compatibility_cost(r, p)
        values.append((f, r))
    return min(values)[1]


def linspace(a: float, b: float, n: int) -> list[float]:
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def rank_color(rank: int, M: int) -> str:
    # A compact viridis-like gradient without external plotting dependencies.
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    t = (rank - 1) / max(1, M - 1)
    pos = t * (len(stops) - 1)
    i = min(len(stops) - 2, int(pos))
    u = pos - i
    rgb = tuple(round((1 - u) * stops[i][k] + u * stops[i + 1][k]) for k in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def make_svg(etas: list[float], lambdas: list[float], phase: list[list[int]], threshold: list[tuple[float, float, int, float]], M: int) -> str:
    width, height = 760, 520
    left, right, top, bottom = 70, 25, 35, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    eta_min, eta_max = etas[0], etas[-1]
    lam_min, lam_max = lambdas[0], lambdas[-1]

    def xmap(eta: float) -> float:
        return left + (eta - eta_min) / (eta_max - eta_min) * plot_w

    def ymap(lam: float) -> float:
        return top + plot_h - (lam - lam_min) / (lam_max - lam_min) * plot_h

    cell_w = plot_w / len(etas)
    cell_h = plot_h / len(lambdas)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;font-size:13px}.small{font-size:11px}.title{font-size:16px;font-weight:bold}</style>',
        f'<text x="{width/2}" y="22" text-anchor="middle" class="title">Rank-sector toy model: M=16, T=1, C(r)=|r-4|</text>',
    ]
    for iy, lam in enumerate(lambdas):
        for ix, eta in enumerate(etas):
            rank = phase[iy][ix]
            x = left + ix * cell_w
            y = top + plot_h - (iy + 1) * cell_h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w+0.2:.2f}" height="{cell_h+0.2:.2f}" fill="{rank_color(rank, M)}"/>')
    # Threshold polyline.
    pts = []
    for eta, lam, _, _ in threshold:
        lam_clipped = max(lam_min, min(lam_max, lam))
        pts.append(f'{xmap(eta):.2f},{ymap(lam_clipped):.2f}')
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="white" stroke-width="3"/>')
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="black" stroke-width="1.2"/>')
    # Axes.
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black" stroke-width="1"/>')
    for eta in [-1, -0.5, 0, 0.5, 1]:
        x = xmap(eta)
        parts.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+5}" stroke="black"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" class="small">{eta:g}</text>')
    for lam in [0, 0.5, 1, 1.5, 2]:
        y = ymap(lam)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{lam:g}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle">energy tilt eta</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 18 {top+plot_h/2})">compatibility strength lambda</text>')
    parts.append(f'<text x="{left+plot_w-8}" y="{top+18}" text-anchor="end" fill="black" class="small">black/white curve: lambda_c(eta)</text>')
    # Minimal legend for rank colors.
    legend_x, legend_y = left + 10, top + 12
    for idx, rank in enumerate([1, 4, 8, 12, 16]):
        x = legend_x + idx * 58
        parts.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{rank_color(rank, M)}" stroke="black" stroke-width="0.3"/>')
        parts.append(f'<text x="{x+20}" y="{legend_y+12}" class="small">r={rank}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    table_rows = []
    for M in [6, 8, 10, 12, 16, 20, 24]:
        lam_c, competitor, numerator = lambda_threshold(M, T=1.0, eta=0.0)
        table_rows.append(
            {
                "M": M,
                "T": 1.0,
                "eta": 0.0,
                "lambda_c_over_T": lam_c,
                "dominant_competitor": competitor,
                "numerator": numerator,
            }
        )

    with (OUT / "epges_rank_sector_threshold_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    M = 16
    etas = linspace(-1.0, 1.0, 161)
    lambdas = linspace(0.0, 2.0, 161)
    phase: list[list[int]] = []
    threshold = []
    for lam in lambdas:
        row = []
        for eta in etas:
            row.append(selected_rank(M, T=1.0, eta=float(eta), lam=float(lam)))
        phase.append(row)
    for eta in etas:
        lam_c, competitor, numerator = lambda_threshold(M, T=1.0, eta=float(eta))
        threshold.append((eta, lam_c, competitor, numerator))

    with (OUT / "epges_rank_sector_threshold_phase.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eta", "lambda_c", "dominant_competitor", "numerator"])
        writer.writerows(threshold)

    svg = make_svg(etas, lambdas, phase, threshold, M)
    (OUT / "epges_rank_sector_threshold_phase.svg").write_text(svg, encoding="utf-8")

    print("wrote", OUT / "epges_rank_sector_threshold_table.csv")
    print("wrote", OUT / "epges_rank_sector_threshold_phase.csv")
    print("wrote", OUT / "epges_rank_sector_threshold_phase.svg")


if __name__ == "__main__":
    main()
