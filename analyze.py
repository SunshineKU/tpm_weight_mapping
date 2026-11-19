"""Exact analysis of bit-to-weight mappings for TPM-based reconciliation.

For every mapping (asymmetric, modulo baseline, all order-preserving
placements, Hamming-weight) and every L, this script computes

  * entropy H_f and the entropy lost in the mapping,
  * the mean weight mu_f (used for mirror removal and tie-breaking in selection),
  * locality: average absolute (Lambda_bar_f) and average squared
    (Lambda_bar_f^(2)) weight change caused by a single bit flip,
  * initial similarity per weight: q_f = P[w_A != w_X], delta1 = E|w_A - w_X|,
    delta2 = E[(w_A - w_X)^2],

where X is Bob (error rate p = QBER) or Eve under three knowledge models
(intercept-resend, optimal individual, uncorrelated).

Blocks are read in natural binary order, most significant bit first.
The network size is fixed (K, N), so every mapping gives KN weights and the
key length is n = b * KN.

Output is a long-format CSV with one row per (mapping, QBER, pair).
"""

import argparse
import csv
from itertools import combinations
from math import comb

import numpy as np


# ---------------------------------------------------------------- mappings

def popcount(x):
    return np.array([bin(int(v)).count("1") for v in np.ravel(x)]).reshape(np.shape(x))


def symmetric_b(L):
    return int(np.ceil(np.log2(2 * L + 1)))


def symmetric_mappings(L):
    """Yield (family, D, f) for the mappings onto the symmetric range {-L..L}."""
    b = symmetric_b(L)
    s = 2**b - (2 * L + 1)
    u = np.arange(2**b)
    values = np.arange(-L, L + 1)

    yield "modulo", "-", u % (2 * L + 1) - L
    yield "hamming", "-", popcount(u) - (b + 1) // 2          # wt(u) - ceil(b/2)
    for D in combinations(values, s):
        f = np.sort(np.concatenate([values, D]))               # f_D(u) = m_u
        yield "placement", ";".join(map(str, D)), f


def asymmetric_mapping(L):
    """Yield the asymmetric mapping onto {-L..L-1} if L is a power of two."""
    if L >= 1 and (L & (L - 1)) == 0:
        b = int(np.log2(L)) + 1
        yield "asym", "-", np.arange(2**b) - L


# -------------------------------------------------------------- properties

def properties(F, b):
    """Entropy, mean and locality (average absolute and average squared
    single-bit-flip change) for a stack of mappings F (rows = mappings)."""
    U = 2**b
    u = np.arange(U)

    w = np.arange(F.min(), F.max() + 1)
    P = np.stack([(F == v).sum(1) for v in w], axis=1) / U
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(P * np.log2(P), axis=1)
    mean = P @ w

    jumps = np.stack([np.abs(F - F[:, u ^ (1 << i)]) for i in range(b)], axis=2)
    lam_avg = jumps.mean(axis=(1, 2))
    lam_sq = (jumps**2).mean(axis=(1, 2))
    return H, mean, lam_avg, lam_sq


def transition_matrices(b, error_rates):
    """T[k, (u, v)] = e_k^dH(u,v) * (1 - e_k)^(b - dH(u,v))."""
    u = np.arange(2**b)
    dH = popcount(u[:, None] ^ u[None, :])
    e = np.asarray(error_rates)[:, None, None]
    return (e**dH * (1 - e) ** (b - dH)).reshape(len(error_rates), -1)


def similarity(F, b, T):
    """Per-weight q_f, delta1, delta2; shape (mappings, error rates)."""
    U = 2**b
    d = np.abs(F[:, :, None] - F[:, None, :]).reshape(len(F), -1).astype(float)
    q = (d > 0) @ T.T / U
    d1 = d @ T.T / U
    d2 = d**2 @ T.T / U
    return q, d1, d2


# ------------------------------------------------------------ eavesdropper

def error_models(p):
    """(pair, error rate) for Alice-Bob and the three Alice-Eve models."""
    models = [("alice_bob", p)]
    if p <= 0.25:                                   # intercept-resend: p = eta / 4
        models.append(("eve_intercept_resend", 0.5 - p))
    models.append(("eve_optimal_individual", 0.5 - np.sqrt(p * (1 - p))))
    models.append(("eve_uncorrelated", 0.5))
    return models


# -------------------------------------------------------------------- main

COLUMNS = [
    "L", "b", "KN", "n", "family", "D",
    "H", "loss_per_weight", "loss_total", "mean", "lambda_avg", "lambda_sq",
    "pair", "qber", "error_rate",
    "q", "delta1", "delta2",
]


def evaluate(L, b, KN, mappings, qbers, writer):
    jobs = [(p, pair, e) for p in qbers for pair, e in error_models(p)]
    T = transition_matrices(b, [e for _, _, e in jobs])

    F = np.stack([f for _, _, f in mappings])
    H, mean, lam_avg, lam_sq = properties(F, b)
    q, d1, d2 = similarity(F, b, T)

    for i, (family, D, _) in enumerate(mappings):
        base = [L, b, KN, b * KN, family, D,
                H[i], b - H[i], KN * (b - H[i]), mean[i], lam_avg[i], lam_sq[i]]
        for j, (p, pair, e) in enumerate(jobs):
            writer.writerow(base + [pair, p, e, q[i, j], d1[i, j], d2[i, j]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=int, nargs="+", default=list(range(2, 9)))
    ap.add_argument("--qber", type=float, nargs="+",
                    default=[round(0.01 * k, 2) for k in range(1, 12)])
    ap.add_argument("--K", type=int, default=3, help="hidden units")
    ap.add_argument("--N", type=int, default=80, help="inputs per hidden unit")
    ap.add_argument("--out", default="mapping_analysis.csv")
    args = ap.parse_args()
    KN = args.K * args.N

    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for L in args.L:
            b = symmetric_b(L)
            s = 2**b - (2 * L + 1)
            print(f"L = {L}: b = {b}, s = {s}, {comb(2 * L + 1, s)} placements")
            evaluate(L, b, KN, list(symmetric_mappings(L)), args.qber, writer)

            asym = list(asymmetric_mapping(L))
            if asym:
                b_asym = int(np.log2(L)) + 1
                print(f"        asymmetric mapping, b = {b_asym}")
                evaluate(L, b_asym, KN, asym, args.qber, writer)

    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
