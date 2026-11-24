"""Statistical evaluation of the simulation results (Sections III-H, IV and V).

Reads the exact mapping properties (analyze.py) and the simulation summaries
(simulate.py) and computes every statistic reported in the paper:

  1. analytical properties of the mappings              (Section IV-A, Table III)
  2. initial distance and number of iterations          (Section IV-B, Table IV, Fig. 1)
  3. number of iterations and communication leakage     (Section IV-C, Table V)
  4. attacker's success                                 (Section IV-D, Table VI)
  5. comparison with the modulo baseline                (Sections IV-D, IV-F)
  6. asymmetric mapping at equal bits per weight        (Section IV-E, Table VII)
  7. recommended mappings                               (Section IV-F, Table VIII, Fig. 2)
  8. comparison with the Slepian-Wolf limit             (Section V)
  9. entropy of the synchronized weights                (Section V), if the
     summary contains the sync_* columns

Statistical tests (Section III-H):

  * Spearman's rank correlation across the simulated mappings, separately for
    every combination of L and QBER. Its significance is tested with a
    two-sided permutation test: exact (all n! permutations) for n <= EXACT_N
    mappings, otherwise with N_PERM random permutations, p = (1 + #)/(1 + N_PERM).
    Values are rounded to ROUND_DEC decimals before ranking, so that values
    that are equal up to floating-point error are treated as ties.
  * Full success rate against the modulo baseline: two-sided Fisher's exact
    test on the success counts.
  * Partial success increase against the modulo baseline: two-sided z-test on
    the difference of the means, with the standard error from the run-to-run
    standard deviations (5000 runs per configuration). The attacker's final
    overlap is tested in the same way if the summary contains overlap_final_sd.
  * Multiple comparisons: Benjamini-Hochberg correction at false discovery
    rate ALPHA within each family of tests. A family is one measure (e.g. the
    full success rate) for one group of mappings compared with the baseline
    (e.g. all placements, the recommended placements, the Hamming-weight
    mapping), over all L, QBER values and attackers; for correlations, one
    measure over the 77 combinations of L and QBER.
  * Sign tests: two-sided binomial test on the number of positive and negative
    coefficients; coefficients that are exactly zero or undefined are excluded.
  * Differences in the number of iterations: two standard errors of the
    difference of the means (no zero counts occur, so the normal approximation
    is adequate with 5000 runs).

For reference, the report also gives the counts obtained with the criteria
of the first version of the paper (two standard errors with unpooled Wald
standard errors, no correction; t approximation for Spearman's coefficient).

Requires NumPy, SciPy and pandas.
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
EXACT_N = 9
N_PERM = 100_000
ROUND_DEC = 12
SEED = 20261124

KEY = ["L", "family", "D", "qber"]
ATTACKERS = [(a, k) for a in ("simple", "geometric")
             for k in ("eve_uncorrelated", "eve_intercept_resend", "eve_optimal_individual")]
SHORT = {"eve_uncorrelated": "uncorrelated", "eve_intercept_resend": "intercept-resend",
         "eve_optimal_individual": "optimal individual"}


# ------------------------------------------------------------------ helpers

class Report:
    def __init__(self, path):
        self.fh = open(path, "w") if path else None

    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        if self.fh:
            self.fh.write(line + "\n")

    def h(self, title):
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)


def pct(x):
    return f"{100 * x:.0f}%"


def rng_str(values, fmt="{:.0f}"):
    v = np.asarray(values, dtype=float)
    return fmt.format(v.min()) + "-" + fmt.format(v.max())


def bh_reject(p, alpha=ALPHA):
    """Benjamini-Hochberg: boolean array of rejected hypotheses (NaN -> False)."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(p.shape, dtype=bool)
    ok = ~np.isnan(p)
    q = p[ok]
    m = q.size
    if m == 0:
        return out
    order = np.argsort(q)
    passed = q[order] <= alpha * np.arange(1, m + 1) / m
    k = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    rej = np.zeros(m, dtype=bool)
    rej[order[:k]] = True
    out[ok] = rej
    return out


def ranks(x):
    return stats.rankdata(np.round(np.asarray(x, dtype=float), ROUND_DEC))


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_t_p(r, n):
    """Two-sided p-value of the t approximation (as in scipy.stats.spearmanr)."""
    if np.isnan(r) or n < 3:
        return np.nan
    if abs(r) >= 1:
        return 0.0
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))


_PERMS = {}


def spearman_perm_p(x, y, rng):
    """Two-sided permutation p-value of Spearman's coefficient."""
    rx, ry = ranks(x), ranks(y)
    n = len(rx)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return np.nan
    a = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    r0 = a @ ((ry - ry.mean()) / np.linalg.norm(ry - ry.mean()))
    b = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())
    tol = 1e-12
    if n <= EXACT_N:
        if n not in _PERMS:
            _PERMS[n] = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
        vals = b[_PERMS[n]] @ a
        return float(np.mean(np.abs(vals) >= abs(r0) - tol))
    count, done = 0, 0
    while done < N_PERM:
        m = min(20_000, N_PERM - done)
        idx = rng.permuted(np.tile(np.arange(n), (m, 1)), axis=1)
        count += int(np.sum(np.abs(b[idx] @ a) >= abs(r0) - tol))
        done += m
    return (1 + count) / (1 + N_PERM)


def sign_test(values):
    """(positive, negative, defined, two-sided p) for a set of coefficients."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    pos, neg = int((v > 0).sum()), int((v < 0).sum())
    p = stats.binomtest(neg, pos + neg).pvalue if pos + neg else np.nan
    return pos, neg, len(v), p


# ------------------------------------------------------------------ data

def load(analysis_path, summary_paths, families):
    A = pd.read_csv(analysis_path, dtype={"D": str})
    S = pd.concat([pd.read_csv(p, dtype={"D": str}) for p in summary_paths], ignore_index=True)
    S = S.drop_duplicates(subset=KEY + ["K", "N", "attack", "pair"], keep="first")
    if families:
        S = S[S.family.isin(families)]
    S["qber"] = S["qber"].round(4)
    A["qber"] = A["qber"].round(4)

    props = A[A.pair == "alice_bob"][KEY + ["b", "H", "H_min", "mean", "lambda_avg",
                                          "lambda_sq", "q", "delta1", "delta2", "KN"]]
    cfg = S.drop_duplicates(KEY)[KEY + ["runs", "runs_capped", "iter_mean", "iter_var",
                                       "leak_comm"]
                                 + [c for c in ("sync_H_sum", "sync_H_pooled", "sync_Hmin_pooled")
                                    if c in S.columns]]
    C = cfg.merge(props, on=KEY, how="left", validate="one_to_one")
    if C.delta1.isna().any():
        raise SystemExit("mapping_analysis.csv lacks properties of some simulated mappings")

    eve = A[A.pair != "alice_bob"][KEY + ["pair", "delta1", "delta2", "q"]].rename(
        columns={"delta1": "ae_delta1", "delta2": "ae_delta2", "q": "ae_q"})
    E = S.merge(eve, on=KEY + ["pair"], how="left", validate="many_to_one")
    E = E.merge(props[KEY + ["b", "H", "lambda_avg"]], on=KEY, how="left")
    return A, C, E


def recommended(A):
    """Recommended placement per L: smallest lambda_avg, ties by lambda_sq.
    Of a mirrored pair (D, -D), which perform identically, the representative
    kept by select_mappings.py is used: the non-positive mean, or the
    lexicographically smaller D if both means are zero."""
    P = A[(A.pair == "alice_bob") & (A.family == "placement")].drop_duplicates(["L", "D"])
    P = P.assign(la=P.lambda_avg.round(9), ls=P.lambda_sq.round(9))

    def representative(row):
        k = tuple(sorted(int(x) for x in row.D.split(";")))
        mk = tuple(sorted(-x for x in k))
        return row["mean"] < -1e-12 or (abs(row["mean"]) <= 1e-12 and k <= mk)

    P = P[P.apply(representative, axis=1)]
    return {L: g.sort_values(["la", "ls", "D"]).iloc[0].D for L, g in P.groupby("L")}


def is_rec(df, rec):
    return (df.family == "placement") & (df.D == df.L.map(rec))


# ------------------------------------------------------------------ sections

def section_properties(A, rep, rec):
    rep.h("1. Analytical properties (Section IV-A, Table III); delta1 at 6% QBER")
    X = A[(A.pair == "alice_bob") & (A.qber == 0.06)]
    rep(f"{'L':>2} | {'H pl':>5} {'Hmin pl':>7} {'H ham':>5} {'H asym':>6} {'H clip':>6} | "
        f"{'Lam mod':>7} {'Lam pl':>11} {'Lam asym':>8} {'Lam clip':>8} | "
        f"{'d1 mod':>6} {'d1 pl':>11} {'d1 ham':>6} {'d1 asym':>7} {'d1 clip':>7}")
    diffs, ham_loss = [], []
    for L, g in X.groupby("L"):
        pl = g[g.family == "placement"]
        get = lambda fam, col: (g[g.family == fam][col].iloc[0]
                                if (g.family == fam).any() else np.nan)
        diffs.append((L, get("modulo", "lambda_avg") - pl.lambda_avg.max()))
        ham_loss.append(pl.H.iloc[0] - get("hamming", "H"))
        rep(f"{L:>2} | {pl.H.iloc[0]:5.2f} {pl.H_min.iloc[0]:7.2f} {get('hamming', 'H'):5.2f} "
            f"{get('asym', 'H'):6.2f} {get('clip', 'H'):6.2f} | "
            f"{get('modulo', 'lambda_avg'):7.2f} {pl.lambda_avg.min():5.2f}-{pl.lambda_avg.max():5.2f} "
            f"{get('asym', 'lambda_avg'):8.2f} {get('clip', 'lambda_avg'):8.2f} | "
            f"{get('modulo', 'delta1'):6.3f} {pl.delta1.min():5.3f}-{pl.delta1.max():5.3f} "
            f"{get('hamming', 'delta1'):6.3f} {get('asym', 'delta1'):7.3f} {get('clip', 'delta1'):7.3f}")
    rep("modulo lambda_avg minus the largest placement lambda_avg:",
        ", ".join(f"L={L}: {d:.2f}" for L, d in diffs))
    KN = int(X.KN.iloc[0])
    rep(f"Hamming entropy loss vs placements: {min(ham_loss):.2f}-{max(ham_loss):.2f} bits/weight, "
        f"{KN * min(ham_loss):.0f}-{KN * max(ham_loss):.0f} bits per key")
    clip = X[X.family == "clip"]
    if len(clip):
        pl = X[X.family == "placement"].groupby("L")
        rep("clip: delta1 larger than every placement at L =",
            [int(L) for L, r in clip.set_index("L").iterrows() if r.delta1 > pl.delta1.max()[L]],
            "| entropy loss vs placements (bits/weight):",
            {int(L): round(pl.H.first()[L] - r.H, 3) for L, r in clip.set_index("L").iterrows()})
    rep("recommended placements:", rec)


def section_correlation(C, rep, rng, rec):
    rep.h("2. Initial distance and number of iterations (Section IV-B, Table IV)")
    measures = ["delta1", "delta2", "q", "lambda_avg", "lambda_sq", "mean"]
    rows = []
    for (L, qb), g in C.groupby(["L", "qber"]):
        r = {"L": L, "qber": qb, "n": len(g)}
        for m in measures:
            r[m] = spearman(g[m], g.iter_mean)
        for m in ("delta1", "delta2", "q"):
            r[m + "_pt"] = spearman_t_p(r[m], len(g))
            r[m + "_pp"] = spearman_perm_p(g[m].values, g.iter_mean.values, rng)
        gp = g[g.family == "placement"]
        r["pl_delta1"] = spearman(gp.delta1, gp.iter_mean)
        r["pl_n"] = len(gp)
        r["pl_delta1_pp"] = spearman_perm_p(gp.delta1.values, gp.iter_mean.values, rng)
        rows.append(r)
    R = pd.DataFrame(rows)
    n_comb = len(R)
    rep(f"{n_comb} combinations of L and QBER; mappings per L:",
        R.groupby("L").n.first().to_dict())
    for m, label in (("delta1", "delta1"), ("delta2", "delta2"), ("q", "q_f"),
                     ("pl_delta1", "delta1, placements only")):
        v = R[m]
        line = (f"{label:<24} median {v.median():.2f}  positive {int((v > 0).sum())}/{n_comb}"
                f"  per-L medians {[round(x, 2) for x in R.groupby('L')[m].median()]}")
        pp = R[m + "_pp"]
        line += (f"\n{'':<24} permutation test: significant {int((pp < ALPHA).sum())}/{n_comb} "
                 f"unadjusted, {int(bh_reject(pp).sum())}/{n_comb} with BH")
        if m + "_pt" in R:
            line += f"; t approximation (first version): {int((R[m + '_pt'] < ALPHA).sum())}/{n_comb}"
        rep(line)
    ns = R[bh_reject(R.delta1_pp) == False]
    if len(ns):
        rep("delta1 not significant (BH) at:",
            [(int(a), b, round(c, 2), round(d, 3)) for a, b, c, d in
             zip(ns.L, ns.qber, ns.delta1, ns.delta1_pp)])
    for m in ("lambda_avg", "lambda_sq", "mean"):
        rep(f"{m:<24} median {R[m].median():.2f}  positive {int((R[m] > 0).sum())}/{n_comb}")
    lowest = R.groupby("L").pl_delta1.median().idxmin()
    rep(f"placements-only coefficient weakest at L = {lowest}")
    for L in sorted(C.L.unique()):
        pl = C[(C.L == L) & (C.family == "placement")]
        spread = pl.groupby("qber").delta1.agg(lambda x: x.max() / x.min() - 1).median()
        rep(f"  L={L}: placements' initial distances differ by {pct(spread)} (median over QBER)")

    # Fig. 1: relative to modulo
    rel = []
    for (L, qb), g in C.groupby(["L", "qber"]):
        mod = g[g.family == "modulo"].iloc[0]
        h = g[~g.family.isin(["modulo", "asym"])]
        rel += list(zip(h.delta1 / mod.delta1, h.iter_mean / mod.iter_mean))
    rel = np.array(rel)
    rep(f"Fig. 1: Pearson r over {len(rel)} symmetric mappings (relative to modulo): "
        f"{np.corrcoef(rel.T)[0, 1]:.3f}")
    asym = C[C.family == "asym"]
    if len(asym):
        mod = C[C.family == "modulo"].set_index(["L", "qber"])
        slope, icpt = np.polyfit(rel[:, 0], rel[:, 1], 1)
        ra = asym.apply(lambda r: (r.delta1 / mod.loc[(r.L, r.qber)].delta1,
                                   r.iter_mean / mod.loc[(r.L, r.qber)].iter_mean), axis=1)
        below = sum(y < slope * x + icpt for x, y in ra)
        rep(f"asymmetric mapping below the fitted line in {below}/{len(ra)} configurations")
    return R


def section_iterations(C, rep, rec):
    rep.h("3. Number of iterations and communication leakage (Section IV-C, Table V)")
    mod = C[C.family == "modulo"].set_index(["L", "qber"])
    C = C.assign(
        red_it=1 - C.iter_mean / C.set_index(["L", "qber"]).index.map(mod.iter_mean),
        red_leak=1 - C.leak_comm / C.set_index(["L", "qber"]).index.map(mod.leak_comm),
        per_it=C.leak_comm / C.iter_mean)
    top = C.qber.max()
    rep(f"Mean number of iterations at {top:.0%} QBER (reduction relative to modulo):")
    for L, g in C[C.qber == top].groupby("L"):
        pl = g[g.family == "placement"]
        parts = [f"modulo {mod.loc[(L, top)].iter_mean:.0f}",
                 f"placements {pl.iter_mean.min():.0f}-{pl.iter_mean.max():.0f} "
                 f"(-{pct(pl.red_it.min())} to -{pct(pl.red_it.max())})"]
        for fam in ("hamming", "asym", "clip"):
            r = g[g.family == fam]
            if len(r):
                parts.append(f"{fam} {r.iter_mean.iloc[0]:.0f} ({-r.red_it.iloc[0]:+.0%})")
        rep(f"  L={L}: " + ", ".join(parts))
    pl = C[C.family == "placement"]
    rec_rows = C[is_rec(C, rec)]
    allmod = (C.groupby(["L", "qber"]).apply(
        lambda g: (g[g.family == "placement"].iter_mean < g[g.family == "modulo"].iter_mean.iloc[0]).all(),
        include_groups=False)).all()
    rep(f"modulo needs more iterations than every placement in every combination: {allmod}")
    rep(f"placements, reduction over all QBER: {rng_str(100 * pl.red_it)}%; "
        f"recommended: {rng_str(100 * rec_rows.red_it)}% (median {100 * rec_rows.red_it.median():.0f}%)")
    spread = pl.groupby(["L", "qber"]).iter_mean.agg(lambda x: x.max() / x.min() - 1)
    rep(f"placements differ among themselves by {rng_str(100 * spread)}% "
        f"(max/min - 1 per combination)")
    for fam in ("hamming", "asym", "clip"):
        r = C[C.family == fam]
        if len(r):
            rep(f"{fam}: reduction {rng_str(100 * r.red_it)}% (iterations), "
                f"{rng_str(100 * r.red_leak)}% (leakage)")
    rep(f"leakage per iteration over all configurations: {C.per_it.min():.2f}-{C.per_it.max():.2f} "
        f"bits (median {C.per_it.median():.2f})")
    rep(f"placements, leakage reduction: {rng_str(100 * pl.red_leak)}%")
    return C


def section_security(E, C, rep, rec):
    rep.h("4. Attacker's success (Section IV-D, Table VI)")

    def corr_block(df, x, y, label):
        rep(label)
        for a, k in ATTACKERS:
            G = df[(df.attack == a) & (df.pair == k)]
            v = [spearman(g[x], g[y]) for _, g in G.groupby(["L", "qber"])]
            pos, neg, n, p = sign_test(v)
            rep(f"  {a:<9} {SHORT[k]:<18} positive {pos:>2}, negative {neg:>2} of {n:>2} defined; "
                f"sign test p = {p:.3g}")

    corr_block(E, "iter_mean", "overlap_increase_mean",
               "iterations vs partial success increase (all mappings)")
    corr_block(E, "iter_mean", "full_success_rate",
               "iterations vs full success rate (all mappings)")
    corr_block(E[E.family == "placement"], "iter_mean", "full_success_rate",
               "iterations vs full success rate (placements)")
    corr_block(E[E.family == "placement"], "ae_delta1", "full_success_rate",
               "Alice-Eve delta1 vs full success rate (placements)")
    corr_block(E, "ae_delta1", "full_success_rate",
               "Alice-Eve delta1 vs full success rate (all mappings)")
    corr_block(E[E.family == "placement"], "ae_delta2", "full_success_rate",
               "Alice-Eve delta2 vs full success rate (placements)")
    corr_block(E, "ae_delta2", "full_success_rate",
               "Alice-Eve delta2 vs full success rate (all mappings)")
    corr_block(E[E.family == "placement"], "ae_delta1", "overlap_increase_mean",
               "Alice-Eve delta1 vs partial success increase (placements)")
    corr_block(E, "ae_delta1", "overlap_increase_mean",
               "Alice-Eve delta1 vs partial success increase (all mappings)")

    top = E.qber.max()
    rep(f"\nTable VI: full success rate (%) of the geometric attack at {top:.0%} QBER, "
        f"optimal individual / uncorrelated")
    G = E[(E.attack == "geometric") & (E.qber == top)]
    for L, g in G.groupby("L"):
        def cell(sel):
            oi = 100 * sel[sel.pair == "eve_optimal_individual"].full_success_rate
            un = 100 * sel[sel.pair == "eve_uncorrelated"].full_success_rate
            if not len(oi):
                return "--"
            if len(oi) == 1:
                return f"{oi.iloc[0]:.1f} / {un.iloc[0]:.1f}"
            return f"{oi.min():.1f}-{oi.max():.1f} / {un.min():.1f}-{un.max():.1f}"
        rep(f"  L={L}: modulo {cell(g[g.family == 'modulo'])} | placements "
            f"{cell(g[g.family == 'placement'])} | hamming {cell(g[g.family == 'hamming'])} | "
            f"asym {cell(g[g.family == 'asym'])} | clip {cell(g[g.family == 'clip'])}")
    s = E[E.attack == "simple"]
    rep(f"simple attack: max full success at L=2 {100 * s[s.L == 2].full_success_rate.max():.1f}%, "
        f"at L>=3 {100 * s[s.L >= 3].full_success_rate.max():.2f}%")
    g1 = E[(E.attack == "geometric") & (E.qber == E.qber.min())]
    rep(f"geometric attack at {E.qber.min():.0%} QBER: max {100 * g1.full_success_rate.max():.1f}%")

    asym = E[(E.family == "asym") & (E.attack == "geometric")]
    if len(asym):
        pl = E[(E.family == "placement") & (E.attack == "geometric")]
        mx = pl.groupby(["L", "qber", "pair"]).overlap_increase_mean.max()
        mo = pl.groupby(["L", "qber", "pair"]).overlap_init_mean.max()
        a = asym.set_index(["L", "qber", "pair"])
        exceed = int((a.overlap_increase_mean > mx.loc[a.index]).sum())
        gap = (a.overlap_init_mean - mo.loc[a.index])
        rep(f"asym, geometric: partial increase above every placement in {exceed}/{len(a)}; "
            f"initial overlap above the largest placement by at most "
            f"{100 * gap[a.index.get_level_values('L') >= 4].max():.1f} pp (L >= 4)")


def compare_with_baseline(E, rep, rec, base="modulo"):
    """Tests of every mapping against the baseline family `base` (modulo or clip)."""
    if base == "modulo":
        rep.h("5. Comparison with the modulo baseline (Sections III-H, IV-D, IV-F)")
    else:
        rep.h(f"5b. Comparison with the {base} mapping (Sections IV-C, IV-F)")
    key = ["L", "qber", "attack", "pair"]
    has_final = "overlap_final_sd" in E.columns and E.overlap_final_sd.notna().all()
    cols = ["runs", "full_success_rate", "overlap_increase_mean", "overlap_increase_sd"]
    if has_final:
        cols += ["overlap_final_mean", "overlap_final_sd"]
    mod = E[E.family == base][key + cols]
    if not len(mod):
        return {}
    X = E[E.family != base].merge(mod, on=key, suffixes=("", "_m"))
    k1 = np.rint(X.full_success_rate * X.runs).astype(int)
    k0 = np.rint(X.full_success_rate_m * X.runs_m).astype(int)
    X["p_full"] = [stats.fisher_exact([[a, n - a], [b, m - b]])[1]
                   for a, n, b, m in zip(k1, X.runs, k0, X.runs_m)]
    X["d_full"] = X.full_success_rate - X.full_success_rate_m
    X["d_inc"] = X.overlap_increase_mean - X.overlap_increase_mean_m
    se_inc = np.sqrt(X.overlap_increase_sd**2 / X.runs + X.overlap_increase_sd_m**2 / X.runs_m)
    X["p_inc"] = 2 * stats.norm.sf(np.abs(X.d_inc / se_inc))
    measures = [("full", "full success rate"), ("inc", "partial success increase")]
    if has_final:
        X["d_fin"] = X.overlap_final_mean - X.overlap_final_mean_m
        se_fin = np.sqrt(X.overlap_final_sd**2 / X.runs + X.overlap_final_sd_m**2 / X.runs_m)
        X["p_fin"] = 2 * stats.norm.sf(np.abs(X.d_fin / se_fin))
        measures.append(("fin", "final overlap"))
    # criteria of the first version: 2 SE, unpooled Wald SE, no correction
    se_full = np.sqrt(X.full_success_rate * (1 - X.full_success_rate) / X.runs
                      + X.full_success_rate_m * (1 - X.full_success_rate_m) / X.runs_m)
    X["old_full_hi"], X["old_full_lo"] = X.d_full > 2 * se_full, X.d_full < -2 * se_full
    X["old_inc_hi"], X["old_inc_lo"] = X.d_inc > 2 * se_inc, X.d_inc < -2 * se_inc

    groups = {"placements (all)": X.family == "placement",
              "recommended": is_rec(X, rec),
              "hamming": X.family == "hamming",
              "asym": X.family == "asym",
              "clip": X.family == "clip"}
    if base != "modulo":
        groups = {"recommended": is_rec(X, rec)}
    out = {}
    for name, sel in groups.items():
        G = X[sel].copy()
        if not len(G):
            continue
        for m, _ in measures:
            G[f"sig_{m}"] = bh_reject(G[f"p_{m}"].values)
            G[f"raw_{m}"] = G[f"p_{m}"] < ALPHA
        out[name] = G
        rep(f"\n{name}: {len(G)} comparisons ({G.groupby(['L', 'qber']).ngroups} "
            f"combinations of L and QBER x {G.groupby(['attack', 'pair']).ngroups} attackers)")
        for m, label in measures:
            hi, lo = G[f"d_{m}"] > 0, G[f"d_{m}"] < 0
            old_hi = f"{int(G[f'old_{m}_hi'].sum()):>4}" if f"old_{m}_hi" in G else "   -"
            old_lo = f"{int(G[f'old_{m}_lo'].sum()):>4}" if f"old_{m}_lo" in G else "   -"
            rep(f"  {label:<25} higher: {int((G[f'sig_{m}'] & hi).sum()):>4} BH, "
                f"{int((G[f'raw_{m}'] & hi).sum()):>4} unadjusted, {old_hi} first version | "
                f"lower: {int((G[f'sig_{m}'] & lo).sum()):>4} BH, "
                f"{int((G[f'raw_{m}'] & lo).sum()):>4} unadjusted, {old_lo} first version | "
                f"lower point estimate: {int(lo.sum())}")
            if m == "fin":
                rep(f"  {'':<25} median difference of the final overlap: "
                    f"{100 * G.d_fin.median():+.2f} percentage points")
        inc_cfg = G.groupby(["L", "qber"]).apply(
            lambda g: ((g.sig_full & (g.d_full > 0)) | (g.sig_inc & (g.d_inc > 0))).any(),
            include_groups=False)
        inc_raw = G.groupby(["L", "qber"]).apply(
            lambda g: ((g.raw_full & (g.d_full > 0)) | (g.raw_inc & (g.d_inc > 0))).any(),
            include_groups=False)
        inc_old = G.groupby(["L", "qber"]).apply(
            lambda g: (g.old_full_hi | g.old_inc_hi).any(), include_groups=False)
        rep(f"  configurations with a significant increase in either measure: "
            f"{int(inc_cfg.sum())} BH, {int(inc_raw.sum())} unadjusted, "
            f"{int(inc_old.sum())} first version, of {len(inc_cfg)}")
        if name in ("placements (all)", "recommended", "clip"):
            hi = G[(G.d_full > 0) & (G.raw_full | G.old_full_hi)]
            for _, r in hi.iterrows():
                rep(f"    full success higher: L={r.L} {r.family} {r.D} qber={r.qber} {r.attack} "
                    f"{SHORT[r.pair]}: {r.full_success_rate * r.runs:.0f} vs "
                    f"{r.full_success_rate_m * r.runs_m:.0f} of {r.runs}, Fisher p = {r.p_full:.3g}, "
                    f"significant with BH: {r.sig_full}")
            hi = G[(G.d_inc > 0) & (G.raw_inc | G.old_inc_hi)]
            if len(hi) > 10:
                rep(f"    partial increase higher in {len(hi)} comparisons (not listed)")
                hi = hi.iloc[:0]
            for _, r in hi.iterrows():
                rep(f"    partial increase higher: L={r.L} {r.family} {r.D} qber={r.qber} {r.attack} "
                    f"{SHORT[r.pair]}: {r.overlap_increase_mean:.5f} vs {r.overlap_increase_mean_m:.5f}, "
                    f"p = {r.p_inc:.3g}, significant with BH: {r.sig_inc}")
    pl = out.get("placements (all)")
    if pl is not None:
        top = E.qber.max()
        g = pl[(pl.qber == top) & (pl.attack == "geometric")
               & (pl.pair.isin(["eve_optimal_individual", "eve_uncorrelated"]))]
        above = g[g.d_full > 0]
        rep(f"Table VI placements above modulo: {len(above)} entries, "
            f"significant (BH): {int(above.sig_full.sum())}, unadjusted: {int(above.raw_full.sum())}")
    X.to_csv(f"comparisons_with_{base}.csv", index=False)
    rep(f"(all tests written to comparisons_with_{base}.csv)")
    return out


def section_asym(C, E, rep, rec):
    asym = C[C.family == "asym"]
    if not len(asym):
        return
    rep.h("6. Asymmetric mapping at equal bits per weight (Section IV-E, Table VII)")
    q0, q1 = C.qber.min(), C.qber.max()
    for L in sorted(asym.L.unique()):
        a = C[(C.family == "asym") & (C.L == L)].set_index("qber")
        s = C[is_rec(C, rec) & (C.L == L - 1)].set_index("qber")
        if not len(s):
            rep(f"L={L}: no symmetric counterpart at L={L - 1}")
            continue
        KN = int(a.KN.iloc[0])
        oi = E[(E.attack == "geometric") & (E.pair == "eve_optimal_individual") & (E.qber == q1)]
        sa = oi[(oi.family == "asym") & (oi.L == L)].full_success_rate.iloc[0]
        ss = oi[is_rec(oi, rec) & (oi.L == L - 1)].full_success_rate.iloc[0]
        rep(f"b={int(a.b.iloc[0])}: asym L={L}: KN H {KN * a.H.iloc[0]:.0f}, KN Hmin {KN * a.H_min.iloc[0]:.0f}, "
            f"iterations {a.iter_mean[q0]:.0f} / {a.iter_mean[q1]:.0f}, success {100 * sa:.1f}%")
        rep(f"      sym  L={L - 1}: KN H {KN * s.H.iloc[0]:.0f}, KN Hmin {KN * s.H_min.iloc[0]:.0f}, "
            f"iterations {s.iter_mean[q0]:.0f} / {s.iter_mean[q1]:.0f}, success {100 * ss:.1f}%; "
            f"symmetric needs {pct(1 - s.iter_mean[q0] / a.iter_mean[q0])} / "
            f"{pct(1 - s.iter_mean[q1] / a.iter_mean[q1])} fewer iterations")


def section_recommended(C, rep, rec):
    rep.h("7. Recommended mappings (Section IV-F, Table VIII, Fig. 2)")
    q0, q1 = C.qber.min(), C.qber.max()
    mod = C[C.family == "modulo"].set_index(["L", "qber"])
    R = C[is_rec(C, rec)].copy()
    R["red_it"] = 1 - R.iter_mean.values / mod.loc[list(zip(R.L, R.qber))].iter_mean.values
    R["red_leak"] = 1 - R.leak_comm.values / mod.loc[list(zip(R.L, R.qber))].leak_comm.values
    for L, g in R.groupby("L"):
        g = g.set_index("qber")
        lam = C[(C.family == "placement") & (C.L == L)]
        rep(f"L={L}: D={rec[L]:<22} fewer iterations {pct(g.red_it[q0])} / {pct(g.red_it[q1])}, "
            f"less leakage {pct(g.red_leak[q0])} / {pct(g.red_leak[q1])}; "
            f"leakage minus iteration reduction {100 * (g.red_leak - g.red_it).min():.0f}-"
            f"{100 * (g.red_leak - g.red_it).max():.0f} pp; "
            f"variation over QBER: iterations {100 * np.ptp(g.red_it):.0f} pp, "
            f"leakage {100 * np.ptp(g.red_leak):.0f} pp")
    rep(f"overall: iterations {rng_str(100 * R.red_it)}% (median {100 * R.red_it.median():.0f}%), "
        f"leakage {rng_str(100 * R.red_leak)}% (median {100 * R.red_leak.median():.0f}%)")
    tied = {}
    for (L, qb), g in C[C.family == "placement"].groupby(["L", "qber"]):
        best = g.loc[g.iter_mean.idxmin()]
        r = g[g.D == rec[L]].iloc[0]
        se = np.sqrt(best.iter_var / best.runs + r.iter_var / r.runs)
        tied.setdefault(L, []).append(r.iter_mean - best.iter_mean <= 2 * se)
    rep("recommended within two standard errors of the best placement:",
        {int(L): f"{sum(v)}/{len(v)}" for L, v in tied.items()})
    clip = C[C.family == "clip"].set_index(["L", "qber"])
    if len(clip):
        J = R.set_index(["L", "qber"]).join(clip[["iter_mean", "iter_var", "runs", "leak_comm", "H",
                                                  "H_min"]], rsuffix="_c", how="inner")
        J["red_it"] = 1 - J.iter_mean / J.iter_mean_c
        J["red_leak"] = 1 - J.leak_comm / J.leak_comm_c
        se = np.sqrt(J.iter_var / J.runs + J.iter_var_c / J.runs_c)
        sig = int(((J.iter_mean_c - J.iter_mean) > 2 * se).sum())
        rep(f"relative to the clipping mapping (L = {sorted(J.index.get_level_values(0).unique())}): "
            f"iterations {rng_str(100 * J.red_it)}% (median {100 * J.red_it.median():.0f}%), "
            f"leakage {rng_str(100 * J.red_leak)}% (median {100 * J.red_leak.median():.0f}%), "
            f"fewer by more than two standard errors in {sig}/{len(J)}; entropy gain "
            f"{rng_str(J.H - J.H_c, '{:.2f}')} bits/weight, min-entropy gain "
            f"{rng_str(J.H_min - J.H_min_c, '{:.2f}')} bits/weight")
        for L, g in J.groupby(level=0):
            rep(f"  L={L}: iterations {rng_str(100 * g.red_it)}%, leakage {rng_str(100 * g.red_leak)}%")


def section_slepian_wolf(C, rep, rec):
    rep.h("8. Comparison with the Slepian-Wolf limit (Section V)")
    h = lambda p: -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    R = C[is_rec(C, rec)].copy()
    R["n"] = R.b * R.KN
    R["sw"] = R.n * h(R.qber)
    R["key_H"] = R.KN * R.H
    R["key_Hmin"] = R.KN * R.H_min
    for L, g in R.groupby("L"):
        g = g.set_index("qber")
        over = g[g.leak_comm > g.key_H].index
        over_min = g[g.leak_comm > g.key_Hmin].index
        q0, q1 = g.index.min(), g.index.max()
        rep(f"L={L}: n={g.n.iloc[0]:.0f}, iterations {g.iter_mean[q0]:.0f}/{g.iter_mean[q1]:.0f}, "
            f"n h(p) {g.sw[q0]:.0f}/{g.sw[q1]:.0f}, ratio (1 bit/iteration) "
            f"{g.iter_mean[q0] / g.sw[q0]:.1f}/{g.iter_mean[q1] / g.sw[q1]:.1f}, "
            f"l_f {g.leak_comm[q0]:.0f}/{g.leak_comm[q1]:.0f}, ratio "
            f"{g.leak_comm[q0] / g.sw[q0]:.1f}/{g.leak_comm[q1] / g.sw[q1]:.1f}; "
            f"l_f > KN H_max from {f'{over.min():.0%}' if len(over) else '-'} "
            f"(KN H_max = {g.key_H.iloc[0]:.0f}); "
            f"l_f > KN H_min from {f'{over_min.min():.0%}' if len(over_min) else '-'} "
            f"(KN H_min = {g.key_Hmin.iloc[0]:.0f})")


def section_sync_entropy(C, rep, rec):
    if "sync_H_sum" not in C.columns or C.sync_H_sum.isna().all():
        return
    rep.h("9. Entropy of the synchronized weights (Section V)")
    D = C.dropna(subset=["sync_H_sum"]).copy()
    D["ratio"] = D.sync_H_sum / (D.KN * D.H)
    D["uniform"] = np.log2(np.where(D.family == "asym", 2 * D.L, 2 * D.L + 1))
    rep("per L and family: initial entropy H_f, synchronized pooled entropy and min-entropy "
        "per weight, log2 of the number of weight values, and sum of marginal entropies "
        "relative to KN H_f (ranges over mappings and QBER)")
    for (L, fam), g in D.groupby(["L", "family"]):
        rep(f"  L={L} {fam:<9}: H_f {rng_str(g.H, '{:.2f}')}, sync H {rng_str(g.sync_H_pooled, '{:.3f}')}, "
            f"sync Hmin {rng_str(g.sync_Hmin_pooled, '{:.2f}')}, log2(values) {g.uniform.iloc[0]:.3f}, "
            f"sum/KN H_f {rng_str(g.ratio, '{:.3f}')}")
    rep(f"sum of marginal entropies below KN H_f in {int((D.ratio < 1).sum())} of {len(D)} configurations")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default="mapping_analysis.csv")
    ap.add_argument("--summary", nargs="+", default=["simulation_summary.csv"],
                    help="one or more simulation summaries; the first occurrence of a "
                         "configuration is used")
    ap.add_argument("--family", nargs="+",
                    help="only these families (default: all in the summaries)")
    ap.add_argument("--out", default="evaluation_report.txt")
    args = ap.parse_args()

    rep = Report(args.out)
    A, C, E = load(args.analysis, args.summary, args.family)
    rec = recommended(A)
    rng = np.random.default_rng(SEED)
    rep(f"configurations: {len(C)}, mappings: {C.groupby(['L', 'family', 'D']).ngroups}, "
        f"families: {sorted(C.family.unique())}, runs capped: {int(C.runs_capped.sum())}")
    section_properties(A, rep, rec)
    section_correlation(C, rep, rng, rec)
    section_iterations(C, rep, rec)
    section_security(E, C, rep, rec)
    compare_with_baseline(E, rep, rec, "modulo")
    compare_with_baseline(E, rep, rec, "clip")
    section_asym(C, E, rep, rec)
    section_recommended(C, rep, rec)
    section_slepian_wolf(C, rep, rec)
    section_sync_entropy(C, rep, rec)
    rep(f"\nreport saved to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
