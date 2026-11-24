"""Select the mappings to be simulated, following Section "Selection of mappings".

For every L:
  * the modulo baseline, the Hamming-weight mapping and (L = 2, 4, 8) the
    asymmetric mapping are always included;
  * the clipping mapping is included for every L where it differs from all
    placements, i.e. where it does not have maximum entropy (all L except
    L = 3 and 7, where it equals the placement D = {-L}). Its rows are written
    after all other rows, so that the row index of every other mapping, from
    which simulate.py derives its random seed, is the same as in the original
    selection;
  * order-preserving placements: mirrored pairs (D and -D) are reduced to one,
    keeping the placement with the non-positive mean;
  * if at most --max-all placements remain, all of them are selected;
  * otherwise, for every distinct value of the average absolute locality
    Lambda_bar_f and for the smallest and largest value of the average squared
    locality Lambda_bar_f^(2), two placements are taken among those sharing that
    value: the one with the mean closest to zero and the one with the largest
    absolute mean.

Ties that remain after the mean are broken deterministically by the placement
string. Output: one row per selected mapping, with the full mapping f(u) for
u = 0 .. 2^b - 1 and the clipping range used in synchronization.
"""

import argparse
import csv

import numpy as np

from analyze import (asymmetric_mapping, properties, symmetric_b,
                              symmetric_mappings)

ROUND = 9   # decimals used to compare locality values


def parse_D(D):
    return tuple(sorted(int(x) for x in D.split(";")))


def remove_mirrors(placements, mean):
    """Keep one placement of each mirrored pair (D, -D): the non-positive mean,
    or the lexicographically smaller D if both means are zero."""
    keys = [parse_D(D) for _, D, _ in placements]
    index = {k: i for i, k in enumerate(keys)}
    keep = []
    for i, k in enumerate(keys):
        j = index[tuple(sorted(-x for x in k))]
        if i == j:                                     # self-mirrored: D = -D
            keep.append(i)
        elif mean[i] < -1e-12 or (abs(mean[i]) <= 1e-12 and k < keys[j]):
            keep.append(i)
    return keep


def pick_by_mean(idx, mean, names):
    """Mean closest to zero and largest absolute mean among idx (may coincide)."""
    order = sorted(idx, key=lambda i: (abs(mean[i]), names[i]))
    return {order[0], sorted(idx, key=lambda i: (-abs(mean[i]), names[i]))[0]}


def select_placements(placements, b, max_all):
    F = np.stack([f for _, _, f in placements])
    _, _, mean, _, _ = properties(F, b)
    keep = remove_mirrors(placements, mean)
    if len(keep) <= max_all:
        return keep, {i: "all" for i in keep}

    _, _, mean, lam_avg, lam_sq = properties(F[keep], b)
    names = [placements[i][1] for i in keep]
    lam_avg = np.round(lam_avg, ROUND)
    lam_sq = np.round(lam_sq, ROUND)
    reasons = {}

    def add(local, reason):
        for i in local:
            reasons.setdefault(i, []).append(reason)

    for v in np.unique(lam_avg):
        add(pick_by_mean(np.flatnonzero(lam_avg == v), mean, names), f"lambda_avg={v:g}")
    for tag, v in (("min", lam_sq.min()), ("max", lam_sq.max())):
        add(pick_by_mean(np.flatnonzero(lam_sq == v), mean, names), f"lambda_sq_{tag}")

    chosen = sorted(reasons)
    return [keep[i] for i in chosen], {keep[i]: "|".join(reasons[i]) for i in chosen}


COLUMNS = ["L", "b", "family", "D", "reason", "w_min", "w_max",
           "H", "mean", "lambda_avg", "lambda_sq", "f"]
# H_min is not written here, so that the columns of the original
# selected_mappings.csv are unchanged; it is in mapping_analysis.csv.


def rows_for(L, b, mappings, reasons, w_min, w_max):
    F = np.stack([f for _, _, f in mappings])
    H, _, mean, lam_avg, lam_sq = properties(F, b)
    for i, (family, D, f) in enumerate(mappings):
        yield [L, b, family, D, reasons[i], w_min, w_max,
               H[i], mean[i], lam_avg[i], lam_sq[i], " ".join(map(str, f))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=int, nargs="+", default=list(range(2, 9)))
    ap.add_argument("--max-all", type=int, default=20,
                    help="simulate all placements if at most this many remain")
    ap.add_argument("--out", default="selected_mappings.csv")
    args = ap.parse_args()

    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for L in args.L:
            b = symmetric_b(L)
            sym = [m for m in symmetric_mappings(L) if m[0] != "clip"]
            fixed = [m for m in sym if m[0] != "placement"]
            placements = [m for m in sym if m[0] == "placement"]

            idx, why = select_placements(placements, b, args.max_all)
            chosen = fixed + [placements[i] for i in idx]
            reasons = ["always"] * len(fixed) + [why[i] for i in idx]
            writer.writerows(rows_for(L, b, chosen, reasons, -L, L))

            asym = list(asymmetric_mapping(L))
            if asym:
                b_asym = int(np.log2(L)) + 1
                writer.writerows(rows_for(L, b_asym, asym, ["always"], -L, L - 1))

            print(f"L = {L}: {len(placements)} placements -> "
                  f"{len(idx)} selected, {len(chosen) + len(asym)} mappings in total")

        # clipping mapping, appended after all other rows (see module docstring)
        for L in args.L:
            b = symmetric_b(L)
            clip = [m for m in symmetric_mappings(L) if m[0] == "clip"]
            placements = [m[2] for m in symmetric_mappings(L) if m[0] == "placement"]
            if any(np.array_equal(clip[0][2], f) for f in placements):
                print(f"L = {L}: clipping mapping equals a placement, not added")
                continue
            writer.writerows(rows_for(L, b, clip, ["always"], -L, L))
            print(f"L = {L}: clipping mapping added")

    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
