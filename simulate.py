"""Monte Carlo simulation of TPM synchronization for the selected mappings.

Reads the mappings chosen by select_mappings.py and, for every mapping and
QBER, runs --runs independent synchronizations (Section "Simulation Setup"):

  * Alice's key is uniformly random; Bob's key flips each bit with prob. p.
  * Both keys are split into b-bit blocks (MSB first) and mapped to KN weights
    with the mapping f; weights are clipped to [w_min, w_max] of the mapping.
  * Alice and Bob update with the Hebbian rule until their weights are
    identical. One iteration = one public input vector + output exchange,
    whether or not an update happens.
  * Six attackers run in parallel and only observe:
    {simple, geometric} x {uncorrelated, intercept-resend, optimal individual}
    initial knowledge. Eve's key flips each of Alice's bits with prob. p_E;
    both attack types with the same knowledge level start from the same key.
  * When Alice and Bob synchronize, each attacker's full success (identical to
    Alice) and overlap (fraction of weights equal to Alice's) are recorded,
    together with her initial overlap.
  * Runs not synchronized within --cap iterations are excluded from the
    averages and counted separately.
  * Entropy of the synchronized weights (Alice's weights when Alice and Bob
    synchronize), estimated across the synchronized runs:
        sync_H_sum     sum over the KN weight positions of the entropy of the
                       weight at that position (plug-in estimate with the
                       Miller-Madow correction), in bits per key. Since the
                       entropy of a vector is at most the sum of the entropies
                       of its components, this is an upper bound on the
                       entropy of the synchronized key (up to estimation error).
        sync_H_pooled  entropy per weight of the distribution of all
                       synchronized weights, pooled over positions and runs
        sync_Hmin_pooled  min-entropy per weight of the same pooled
                       distribution, -log2 max_w P(w)
    Recording them does not draw random numbers, so the other columns are
    identical to those of the original version of this script.

Communication leakage l_f (Section "Simulation Setup", paragraph
"Communication Leakage"), an upper bound on the information that the public
transcript reveals, by the chain rule summed over iterations (inputs are
independent of the keys and leak nothing):
    leak_tau    1 bit per iteration for Alice's output: sum_t P(T >= t),
                i.e. the mean number of iterations
    leak_flag   Bob's output given Alice's, i.e. whether they agree: at most
                h(eps_t) per iteration, eps_t = P(sigma_A != sigma_B | still
                running at t), estimated across runs. Conditioning on less of
                the past only increases entropy, so this is an upper bound.
                Iterations reached by fewer than MIN_ACTIVE runs are charged
                1 bit (2 bits in total with leak_tau).
    leak_stop   H(T), plug-in entropy of the number of iterations with the
                Miller-Madow correction
    leak_comm   l_f = leak_tau + leak_flag + leak_stop

Runs of one configuration are simulated together as numpy arrays, in chunks
of --batch runs to limit memory (about 8*runs*K*N bytes per chunk);
configurations are distributed over --workers processes. With --batch >=
--runs (default) all runs of a configuration share one random stream, as in
the results reported in the paper. Results are appended to the summary file
as configurations finish, and configurations already present in it are
skipped, so an interrupted run can be resumed.
"""

import argparse
import csv
import gzip
import os
import time
from multiprocessing import Pool

import numpy as np

ATTACKS = ("simple", "geometric")
KNOWLEDGE = ("eve_uncorrelated", "eve_intercept_resend", "eve_optimal_individual")
ATTACKERS = [(a, k) for a in ATTACKS for k in KNOWLEDGE]

MIN_ACTIVE = 30          # below this many running runs, charge the worst case


def eve_error_rate(knowledge, p):
    if knowledge == "eve_uncorrelated":
        return 0.5
    if knowledge == "eve_intercept_resend":
        return 0.5 - p
    return 0.5 - np.sqrt(p * (1 - p))


def h2(q):
    """Binary entropy in bits, h(0) = h(1) = 0."""
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
    return np.nan_to_num(out)


# ------------------------------------------------------------- key -> weights

def keys_to_weights(bits, f, b):
    """bits: (..., KN*b) array of 0/1 -> (..., KN) weights, blocks read MSB first."""
    blocks = bits.reshape(*bits.shape[:-1], -1, b)
    u = blocks @ (1 << np.arange(b - 1, -1, -1))
    return f[u]


# ---------------------------------------------------------------- simulation

def simulate(f, b, w_min, w_max, p, K, N, runs, cap, rng):
    """Simulate `runs` synchronizations of one configuration (one batch).

    Returns iterations (runs,), -1 if capped; for every attacker (runs, A)
    arrays: initial overlap, final overlap, full success; the per-iteration
    counts of running runs and of runs in which Alice's and Bob's outputs
    differ (for the communication leakage); and Alice's weights at
    synchronization (runs, K, N), zero for capped runs."""
    KN = K * N
    nA = len(ATTACKERS)
    A = rng.random((runs, KN * b)) < 0.5
    B = A ^ (rng.random(A.shape) < p)
    eve_keys = {k: A ^ (rng.random(A.shape) < eve_error_rate(k, p)) for k in KNOWLEDGE}
    keys = [A, B] + [eve_keys[k] for _, k in ATTACKERS]

    # W[m, r, k, n]: machine m (0 Alice, 1 Bob, 2.. attackers), run r
    W = np.stack([keys_to_weights(key.astype(np.int64), f, b) for key in keys])
    W = W.reshape(len(keys), runs, K, N).astype(np.int8)
    del A, B, keys, eve_keys
    M = W.shape[0]
    geo = np.flatnonzero([a == "geometric" for a, _ in ATTACKERS]) + 2

    iterations = np.full(runs, -1, dtype=np.int64)
    overlap_init = (W[2:] == W[0]).mean(axis=(2, 3)).T                 # (runs, A)
    overlap_final = np.full((runs, nA), np.nan)
    full = np.zeros((runs, nA), dtype=bool)
    W_sync = np.zeros((runs, K, N), dtype=np.int8)
    n_active, n_disagree = [], []      # per iteration: running runs, sigma_A != sigma_B

    active = np.arange(runs)                       # original index of each column

    def finish(done, t):
        idx = active[done]
        iterations[idx] = t
        eq = W[2:, done] == W[0, done][None]                          # (A, d, K, N)
        overlap_final[idx] = eq.mean(axis=(2, 3)).T
        full[idx] = eq.all(axis=(2, 3)).T
        W_sync[idx] = W[0, done]

    done = (W[0] == W[1]).all(axis=(1, 2))
    if done.any():
        finish(done, 0)
        W, active = W[:, ~done], active[~done]

    t = 0
    while active.size and t < cap:
        t += 1
        R = active.size
        x = (rng.integers(0, 2, (R, K, N), dtype=np.int8) << 1) - 1          # +-1
        h = (W * x[None]).sum(axis=3, dtype=np.int32)                       # (M, R, K)
        tau = np.where(h > 0, 1, -1).astype(np.int8)                        # sgn(0) = -1
        sigma = tau.prod(axis=2, dtype=np.int8)                             # (M, R)
        agree = sigma[0] == sigma[1]
        n_active.append(R)
        n_disagree.append(int((~agree).sum()))

        # geometric attack: if sigma_E != sigma_A, flip the hidden unit with the
        # smallest |local field|, which makes sigma_E equal to sigma_A
        wrong = sigma[geo] != sigma[0][None]                                # (g, R)
        if wrong.any():
            k_min = np.abs(h[geo]).argmin(axis=2)                           # (g, R)
            gi, ri = np.nonzero(wrong)
            tau[geo[gi], ri, k_min[gi, ri]] *= -1

        # update only if sigma_A == sigma_B; Alice and Bob use their own output,
        # attackers use Alice's output (Hebbian rule, units with tau_k = sigma)
        target = np.empty((M, R), dtype=np.int8)
        target[0], target[1] = sigma[0], sigma[1]
        target[2:] = sigma[0][None]
        move = (tau == target[:, :, None]) & agree[None, :, None]
        W += x[None] * (tau * move)[..., None]
        np.clip(W, w_min, w_max, out=W)

        done = (W[0] == W[1]).all(axis=(1, 2))
        if done.any():
            finish(done, t)
            W, active = W[:, ~done], active[~done]

    counts = (np.array(n_active, dtype=np.int64), np.array(n_disagree, dtype=np.int64))
    return iterations, overlap_init, overlap_final, full, counts, W_sync


def add_padded(a, b):
    """Elementwise sum of two arrays along axis 0, padding the shorter with 0."""
    if len(a) < len(b):
        a, b = b, a
    out = a.copy()
    out[:len(b)] += b
    return out


def simulate_batched(f, b, w_min, w_max, p, K, N, runs, cap, seq, batch):
    """Run `runs` synchronizations in chunks of `batch` and merge the results.
    With a single chunk, all runs share one random stream."""
    sizes = [min(batch, runs - s) for s in range(0, runs, batch)]
    rngs = ([np.random.default_rng(seq)] if len(sizes) == 1
            else [np.random.default_rng(s) for s in seq.spawn(len(sizes))])
    parts = [simulate(f, b, w_min, w_max, p, K, N, n, cap, r)
             for n, r in zip(sizes, rngs)]
    it = np.concatenate([q[0] for q in parts])
    o_init = np.concatenate([q[1] for q in parts])
    o_final = np.concatenate([q[2] for q in parts])
    full = np.concatenate([q[3] for q in parts])
    W_sync = np.concatenate([q[5] for q in parts])
    n_active, n_disagree = parts[0][4]
    for q in parts[1:]:
        n_active = add_padded(n_active, q[4][0])
        n_disagree = add_padded(n_disagree, q[4][1])
    return it, o_init, o_final, full, (n_active, n_disagree), W_sync


# ------------------------------------------------------------ leakage terms

def communication_leakage(runs, its, counts):
    """leak_tau, leak_flag, leak_stop and leak_comm = l_f (module docstring)."""
    na, nd = counts
    leak_tau = its.mean() if its.size else np.nan
    if na.size:
        hq = np.where(na >= MIN_ACTIVE, h2(nd / np.maximum(na, 1)), 1.0)
        leak_flag = float((na / runs * hq).sum())
    else:
        leak_flag = 0.0
    if its.size:
        _, c = np.unique(its, return_counts=True)
        pr = c / its.size
        leak_stop = float(-(pr * np.log2(pr)).sum()
                          + (len(c) - 1) / (2 * its.size * np.log(2)))
    else:
        leak_stop = np.nan
    return [leak_tau, leak_flag, leak_stop, leak_tau + leak_flag + leak_stop]


def sync_entropy(W_sync, w_min, w_max):
    """sync_H_sum, sync_H_pooled, sync_Hmin_pooled (module docstring) from the
    synchronized weights of the synchronized runs, shape (runs, K, N)."""
    n = W_sync.shape[0]
    if n == 0:
        return [np.nan, np.nan, np.nan]
    X = W_sync.reshape(n, -1).astype(np.int64) - w_min               # (runs, KN)
    V = w_max - w_min + 1
    C = np.stack([(X == v).sum(axis=0) for v in range(V)])           # (V, KN)
    P = C / n
    with np.errstate(divide="ignore", invalid="ignore"):
        H_pos = -np.nansum(P * np.log2(P), axis=0)
    H_pos += ((C > 0).sum(axis=0) - 1) / (2 * n * np.log(2))         # Miller-Madow
    pooled = C.sum(axis=1) / C.sum()
    pooled = pooled[pooled > 0]
    return [float(H_pos.sum()), float(-(pooled * np.log2(pooled)).sum()),
            float(-np.log2(pooled.max()))]


# ------------------------------------------------------------------ driver

SUMMARY = ["L", "b", "family", "D", "K", "N", "qber", "runs", "runs_capped",
           "iter_mean", "iter_var",
           "attack", "pair", "error_rate",
           "full_success_rate", "overlap_init_mean", "overlap_final_mean",
           "overlap_increase_mean", "overlap_increase_sd", "overlap_final_sd",
           # identical in all rows of a configuration
           "leak_tau", "leak_flag", "leak_stop", "leak_comm",
           "sync_H_sum", "sync_H_pooled", "sync_Hmin_pooled"]

RUNS_HEADER = (["L", "family", "D", "K", "N", "qber", "run", "iterations"]
               + [f"{a}_{k}_{m}" for a, k in ATTACKERS for m in ("init", "final", "full")])


def config_key(L, family, D, K, N, qber):
    return (int(L), family, D, int(K), int(N), f"{float(qber):.4f}")


def run_config(job):
    m, qi, p, args = job
    N, K = args.N, args.K
    # N = 80, the size used in the paper, keeps its original seeds
    entropy = [args.seed, m["row"], qi] if N == 80 else [args.seed, m["row"], qi, N]
    seq = np.random.SeedSequence(entropy)
    f = np.array([int(v) for v in m["f"].split()], dtype=np.int64)
    b = int(m["b"])

    t0 = time.time()
    it, o_init, o_final, full, counts, W_sync = simulate_batched(
        f, b, int(m["w_min"]), int(m["w_max"]), p, K, N, args.runs, args.cap,
        seq, args.batch or args.runs)
    ok = it >= 0
    its = it[ok].astype(float)
    leak_cols = communication_leakage(args.runs, its, counts)
    leak_cols += sync_entropy(W_sync[ok], int(m["w_min"]), int(m["w_max"]))

    base = [m["L"], b, m["family"], m["D"], K, N, p, int(ok.sum()), int((~ok).sum()),
            its.mean() if ok.any() else np.nan,
            its.var(ddof=1) if ok.sum() > 1 else np.nan]
    summary = []
    for j, (attack, know) in enumerate(ATTACKERS):
        inc = o_final[ok, j] - o_init[ok, j]
        summary.append(base + [
            attack, know, eve_error_rate(know, p),
            full[ok, j].mean() if ok.any() else np.nan,
            o_init[ok, j].mean() if ok.any() else np.nan,
            o_final[ok, j].mean() if ok.any() else np.nan,
            inc.mean() if ok.any() else np.nan,
            inc.std(ddof=1) if ok.sum() > 1 else np.nan,
            o_final[ok, j].std(ddof=1) if ok.sum() > 1 else np.nan,
        ] + leak_cols)

    per_run = None
    if args.runs_out:
        per_run = []
        for r in range(args.runs):
            row = [m["L"], m["family"], m["D"], K, N, p, r, int(it[r])]
            for j in range(len(ATTACKERS)):
                row += [round(float(o_init[r, j]), 6),
                        round(float(o_final[r, j]), 6) if ok[r] else "",
                        int(full[r, j]) if ok[r] else ""]
            per_run.append(row)

    return config_key(m["L"], m["family"], m["D"], K, N, p), summary, per_run, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mappings", default="selected_mappings.csv")
    ap.add_argument("--L", type=int, nargs="+", help="only these L (default: all)")
    ap.add_argument("--family", nargs="+",
                    help="only these families: modulo, hamming, placement, asym, clip (default: all)")
    ap.add_argument("--qber", type=float, nargs="+",
                    default=[round(0.01 * k, 2) for k in range(1, 12)])
    ap.add_argument("--runs", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=0,
                    help="runs simulated at once (default: all); lower to save memory")
    ap.add_argument("--cap", type=int, default=10**6)
    ap.add_argument("--K", type=int, default=3, help="hidden units")
    ap.add_argument("--N", type=int, default=80, help="inputs per hidden unit")
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="simulation_summary.csv")
    ap.add_argument("--runs-out", help="optional gzipped CSV with every single run")
    args = ap.parse_args()

    with open(args.mappings, newline="") as fh:
        mappings = list(csv.DictReader(fh))
    for i, m in enumerate(mappings):
        m["row"] = i                                     # stable seed per mapping
    mappings = [m for m in mappings
                if (not args.L or int(m["L"]) in args.L)
                and (not args.family or m["family"] in args.family)]
    if not mappings:
        raise SystemExit("no mappings match the selection")

    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != SUMMARY:
                raise SystemExit(f"{args.out} has different columns; choose a new --out file")
            done = {config_key(r["L"], r["family"], r["D"], r["K"], r["N"], r["qber"])
                    for r in reader}
    jobs = [(m, qi, p, args) for m in mappings for qi, p in enumerate(args.qber)
            if config_key(m["L"], m["family"], m["D"], args.K, args.N, p) not in done]
    # longest configurations (large L, QBER) first for better load balance
    jobs.sort(key=lambda j: (-int(j[0]["L"]), -j[2]))
    print(f"{len(jobs)} configurations to run ({len(done)} already done), "
          f"{args.runs} runs each, {args.workers} workers")

    new = not os.path.exists(args.out)
    runs_new = bool(args.runs_out) and not os.path.exists(args.runs_out)
    with open(args.out, "a", newline="") as fh, \
            (gzip.open(args.runs_out, "at", newline="") if args.runs_out else open(os.devnull, "w")) as rh, \
            Pool(args.workers) as pool:
        writer, run_writer = csv.writer(fh), csv.writer(rh)
        if new:
            writer.writerow(SUMMARY)
        if runs_new:
            run_writer.writerow(RUNS_HEADER)
        t_start = time.time()
        for n, (key, summary, per_run, dt) in enumerate(
                pool.imap_unordered(run_config, jobs), 1):
            writer.writerows(summary)
            fh.flush()
            if per_run:
                run_writer.writerows(per_run)
            s = dict(zip(SUMMARY, summary[0]))
            print(f"[{n}/{len(jobs)}] L={key[0]} {key[1]} {key[2]} "
                  f"qber={key[5]}: mean {s['iter_mean']:.0f} it, "
                  f"{s['runs_capped']} capped, leak {s['leak_comm']:.0f}, {dt:.1f}s "
                  f"(elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)

    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
