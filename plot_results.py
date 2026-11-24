"""Figures of the paper.

  fig_mappings_L2.pdf          how the bit patterns of one block are assigned to
                               weights at L = 2 by the modulo, the recommended,
                               the clipping and the asymmetric mapping
                               (Section III-C)
  fig_distance_iterations.pdf  mean iterations vs. initial distance delta1, both
                               relative to the modulo mapping at the same L and
                               QBER, all L and QBER pooled (Section IV-B)
  fig_reduction.pdf            iteration reduction of the recommended placement
                               relative to the modulo mapping vs. QBER, one panel
                               per L, with 95% confidence bands (Section IV-F)

The recommended placement of every L is selected by evaluate.recommended(),
i.e. by the same rule as in the paper and in evaluate.py.

Requires NumPy, SciPy, pandas and Matplotlib.

Usage:
  python plot_results.py --analysis mapping_analysis.csv \\
                         --simulation simulation_summary.csv --outdir figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze import asymmetric_mapping, clip_mapping, symmetric_b, symmetric_mappings
from evaluate import recommended

# One fixed colour and marker per mapping family, the same in every figure
# (categorical slots 1-5 of the palette, in fixed order). Markers are a second
# channel, so the families stay distinguishable in grayscale and for colour-
# vision deficiencies.
FAMILY = {
    "placement": dict(color="#2a78d6", marker="o", label="Placements"),
    "modulo":    dict(color="#eb6834", marker="s", label="Modulo"),
    "hamming":   dict(color="#1baf7a", marker="^", label="Hamming weight"),
    "asym":      dict(color="#eda100", marker="D", label="Asymmetric"),
    "clip":      dict(color="#e87ba4", marker="v", label="Clipping"),
}
INK = "#3d3d3a"
MUTED = "#8a8a84"
GRID = "#e4e3de"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.5, "lines.markersize": 5,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
COLUMN = 3.5             # single IEEE column, inches


def load(analysis_path, simulation_path):
    a = pd.read_csv(analysis_path, dtype={"D": str})
    s = pd.read_csv(simulation_path, dtype={"D": str})
    for df in (a, s):
        df["qber"] = df["qber"].astype(float).round(4)
    key = ["L", "family", "D", "qber"]
    it = s[key + ["runs", "iter_mean", "iter_var"]].drop_duplicates(key)
    ab = a[a["pair"] == "alice_bob"][key + ["KN", "H", "delta1"]]
    d = it.merge(ab, on=key, validate="one_to_one")
    return a, d


# ------------------------------------------------------------------ figures

def fig_mappings(rec_D, out, L=2):
    """Bipartite diagram: bit patterns (left) and weights (right) of one block.
    Lines to weight values that receive more than one pattern are drawn darker,
    and these values are set in bold."""
    b = symmetric_b(L)
    sym = list(symmetric_mappings(L))
    modulo = next(f for fam, D, f in sym if fam == "modulo")
    rec_f = next(f for fam, D, f in sym if fam == "placement" and D == rec_D)
    panels = [("Modulo", modulo, b), ("Recommended", rec_f, b),
              ("Clipping", clip_mapping(L), b)]
    asym = list(asymmetric_mapping(L))
    if asym:
        f = asym[0][2]
        panels.append(("Asymmetric", f, int(np.log2(len(f)))))

    fig, axes = plt.subplots(1, len(panels), figsize=(COLUMN, 1.75),
                             gridspec_kw=dict(wspace=0.15))
    for ax, (title, f, bits) in zip(axes, panels):
        f = np.asarray(f)
        n = len(f)
        wmin, wmax = -L, L
        ys_u = np.linspace(1, 0, n)                       # patterns, top to bottom
        weights = np.arange(wmin, wmax + 1)
        ys_w = np.linspace(1, 0, len(weights))            # weights -L (top) ... L
        counts = {w: int((f == w).sum()) for w in weights}
        for u in range(n):
            w = int(f[u])
            yw = ys_w[w - wmin]
            shared = counts[w] > 1
            ax.plot([0.32, 0.66], [ys_u[u], yw], color=INK if shared else MUTED,
                    linewidth=1.1 if shared else 0.7, zorder=1)
        for u in range(n):
            ax.text(0.28, ys_u[u], format(u, f"0{bits}b"), ha="right", va="center",
                    fontsize=6.5, family="monospace")
        for w, y in zip(weights, ys_w):
            used = counts[w] > 0
            label = "0" if w == 0 else ("+" if w > 0 else "\u2212") + str(abs(w))
            ax.text(0.73, y, label, ha="left", va="center", fontsize=6.5,
                    color=INK if used else MUTED,
                    fontweight="bold" if counts[w] > 1 else "normal")
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.08, 1.08)
        ax.axis("off")
    fig.savefig(out)
    plt.close(fig)


def fig_distance(d, out):
    """Iterations vs. initial distance, both relative to the modulo mapping at the
    same L and QBER, pooled over all L and QBER."""
    mod = d[d["family"] == "modulo"][["L", "qber", "delta1", "iter_mean"]] \
        .rename(columns={"delta1": "d_mod", "iter_mean": "i_mod"})
    g = d.merge(mod, on=["L", "qber"])
    g["x"] = g["delta1"] / g["d_mod"]
    g["y"] = g["iter_mean"] / g["i_mod"]

    fig, ax = plt.subplots(figsize=(COLUMN, 2.5))
    for fam in ("placement", "hamming", "clip", "asym", "modulo"):
        st = FAMILY[fam]
        x = g[g["family"] == fam]
        if not len(x):
            continue
        big = fam in ("modulo", "clip", "asym")
        ax.scatter(x["x"], x["y"], s=16 if big else 10, c=st["color"],
                   marker=st["marker"], alpha=1.0 if fam == "modulo" else 0.75,
                   edgecolors="white", linewidths=0.4 if big else 0,
                   label=st["label"], zorder=3 if big else 2)
    sym = g[~g["family"].isin(["modulo", "asym"])]
    print(f"distance figure: Pearson r (symmetric mappings except modulo) = "
          f"{stats.pearsonr(sym['x'], sym['y'])[0]:.3f}, n = {len(sym)}")
    ax.set_xlabel(r"Initial distance $\delta_f^{(1)}$ relative to modulo")
    ax.set_ylabel("Mean iterations relative to modulo")
    ax.legend(loc="lower right", markerscale=1.4, handletextpad=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_reduction(d, rec, out):
    """Reduction of the mean number of iterations of the recommended placement
    relative to the modulo mapping, one panel per L, with 95% confidence bands
    (delta method on the ratio of the two means)."""
    Ls = sorted(rec)
    ncol = 4
    nrow = int(np.ceil(len(Ls) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(COLUMN, 1.15 * nrow + 0.35),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.12, hspace=0.45))
    axes = np.atleast_2d(axes)
    st = FAMILY["placement"]
    q_all = np.sort(d["qber"].unique())
    rows = []
    for ax, L in zip(axes.flat, Ls):
        g = d[d["L"] == L]
        m = g[g["family"] == "modulo"].set_index("qber").sort_index()
        r = g[(g["family"] == "placement") & (g["D"] == rec[L])].set_index("qber").sort_index()
        ratio = r["iter_mean"] / m["iter_mean"]
        se = ratio * np.sqrt(r["iter_var"] / r["runs"] / r["iter_mean"] ** 2
                             + m["iter_var"] / m["runs"] / m["iter_mean"] ** 2)
        red, half = 100 * (1 - ratio), 100 * 1.96 * se
        rows.append((L, float(half.max())))
        q = 100 * red.index.values
        ax.fill_between(q, red - half, red + half, color=st["color"], alpha=0.22,
                        linewidth=0)
        ax.plot(q, red, color=st["color"], marker=st["marker"], markersize=3,
                markeredgecolor="white", markeredgewidth=0.4, linewidth=1.2)
        ax.set_title(f"$L = {L}$", pad=2)
        ax.set_ylim(0, 30)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_xticks([1, 6, 11])
    for ax in axes.flat[len(Ls):]:
        ax.axis("off")
    for j in range(ncol):
        col = [axes[i, j] for i in range(nrow) if i * ncol + j < len(Ls)]
        col[-1].tick_params(labelbottom=True)
    fig.supylabel("Fewer iterations than modulo (%)", fontsize=8, x=0.02)
    fig.supxlabel("QBER (%)", fontsize=8, y=-0.05)
    print("reduction figure: largest half-width of the 95% band per L:",
          {L: round(h, 2) for L, h in rows}, "percentage points")
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default="mapping_analysis.csv")
    ap.add_argument("--simulation", default="simulation_summary.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    a, d = load(args.analysis, args.simulation)
    rec = recommended(a)
    fig_mappings(rec[2], os.path.join(args.outdir, "fig_mappings_L2.pdf"))
    fig_distance(d, os.path.join(args.outdir, "fig_distance_iterations.pdf"))
    fig_reduction(d, rec, os.path.join(args.outdir, "fig_reduction.pdf"))
    print(f"figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()
