# tpm_weight_mapping

Code for the paper **"Bit-to-Weight Mapping for Tree Parity Machine-Based Information Reconciliation in Quantum Key Distribution"**.

In TPM-based information reconciliation for quantum key distribution (QKD), Alice and Bob initialize the weights of their tree parity machines (TPMs) from their sifted keys and synchronize them over a public channel. Before synchronization, blocks of $b$ key bits must be mapped to integer weights in $\{-L, \dots, L\}$. Since $2^b$ bit patterns cannot be assigned one-to-one to the odd number $2L+1$ of weight values, some weight values must receive two patterns. This choice determines how far apart Alice's and Bob's initial weights are, and with it the number of synchronization iterations and the information revealed during synchronization.

This repository contains the five scripts that produce all results and figures of the paper, and their output files with the results reported in the paper:

| Script | Paper section | What it does |
|---|---|---|
| `analyze.py` | III-C – III-E | Exact analysis of every mapping: entropy, min-entropy, locality and initial Alice–Bob and Alice–Eve similarity |
| `select_mappings.py` | III-F | Selects the representative subset of mappings to simulate |
| `simulate.py` | III-G | Monte Carlo simulation of TPM synchronization, the simple and geometric attacks, the communication leakage and the entropy of the synchronized weights |
| `evaluate.py` | III-H, IV, V | Statistical evaluation: every statistic, table entry and test reported in the paper |
| `plot_results.py` | III-C, IV-B, IV-F | Figures of the paper (`fig_mappings_L2.pdf`, `fig_distance_iterations.pdf`, `fig_reduction.pdf`) |

| Archive | Contains | Produced by | Contents |
|---|---|---|---|
| `mapping_analysis.zip` | `mapping_analysis.csv` | `analyze.py` | Exact properties of all mappings for every $L$ and QBER (42,944 rows) |
| `selected_mappings.zip` | `selected_mappings.csv` | `select_mappings.py` | The 121 mappings simulated in the paper |
| `simulation_summary.zip` | `simulation_summary.csv` | `simulate.py` | Simulation results for all 1331 configurations, 5000 runs each (7986 rows) |
| `evaluation.zip` | `evaluation_report.txt`, `comparisons_with_modulo.csv`, `comparisons_with_clip.csv` | `evaluate.py` | Report with all statistics of the paper, and every single test against the two baselines |

The result files are compressed. Extract them into the repository directory before use:

```bash
unzip mapping_analysis.zip
unzip selected_mappings.zip
unzip simulation_summary.zip
```

The column formats are described under [Output files](#output-files).

## Requirements

- Python 3
- NumPy (all scripts)
- SciPy and pandas (`evaluate.py` and `plot_results.py`)
- Matplotlib (`plot_results.py` only)

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0 and Matplotlib 3.11.

```bash
pip install numpy scipy pandas matplotlib
```

## Usage

Run the five scripts in order from the repository directory. The default arguments reproduce the configuration of the paper: $K = 3$, $N = 80$, $L = 2, \dots, 8$, QBER 1–11% in steps of 1%, and 5000 runs per configuration.

```bash
# 1. Exact properties of all mappings -> mapping_analysis.csv
python analyze.py

# 2. Mappings to simulate -> selected_mappings.csv
python select_mappings.py

# 3. Monte Carlo simulation -> new_summary.csv
python simulate.py --out new_summary.csv

# 4. Statistical evaluation -> evaluation_report.txt, comparisons_with_*.csv
python evaluate.py --summary new_summary.csv

# 5. Figures -> figures/*.pdf
python plot_results.py --simulation new_summary.csv
```

Steps 1 and 2 regenerate the files in `mapping_analysis.zip` and `selected_mappings.zip` with identical results (and overwrite them if already extracted). In step 3, use a new `--out` file: `simulate.py` skips every configuration that is already in its output file, so with the default `--out simulation_summary.csv` and the included results extracted, it finds all configurations done and runs nothing. Step 4 reads `mapping_analysis.csv` and the summary given with `--summary` (default `simulation_summary.csv`), and writes the report shown in `evaluation.zip`.

Steps 1, 2, 4 and 5 take seconds to a minute. Step 3 runs 1331 configurations (121 mappings × 11 QBER values), each with 5000 runs, and is by far the most expensive step. Configurations are distributed over all CPU cores by default (`--workers`), and the longest ones (large $L$ and QBER) run first.

`simulate.py` appends each configuration to the summary file as soon as it finishes, and skips configurations already in the file. An interrupted run can therefore be resumed by starting the same command again. The file must have been written by the current version of `simulate.py`; a file with different columns is refused.

### Smaller runs

```bash
# only L = 2 and 3, fewer runs
python simulate.py --L 2 3 --runs 500 --out test_summary.csv

# only some families of mappings
python simulate.py --family modulo placement --out families.csv

# only the clipping mapping
python simulate.py --family clip --out clip.csv

# limit memory by simulating the runs of a configuration in chunks
python simulate.py --batch 1000 --out batched.csv
```

**Note on reproducibility.** The random seed of each configuration is derived from `--seed`, the row of the mapping in `selected_mappings.csv` and the *position* of the QBER in the `--qber` list. Filtering with `--L` or `--family` keeps the seeds, but a shorter `--qber` list changes them. Using `--batch` smaller than `--runs` also changes the random streams. To reproduce the included results exactly, keep the default `--qber` list and do not use `--batch`. For example, `python simulate.py --L 2 --out check.csv` reproduces the rows of `simulation_summary.csv` with $L = 2$.

**Changes from the first version.** The clipping mapping (`clip`) was added. Its rows are written at the end of `selected_mappings.csv`, so the rows, and hence the random seeds, of all other mappings are unchanged. `simulate.py` additionally records the standard deviation of the attacker's final overlap and the entropy of the synchronized weights; recording them draws no random numbers, so all other columns are identical to those of the first version. `analyze.py` additionally writes the min-entropy `H_min`. `evaluate.py` is new; it reproduces the statistics of the first version with its original criteria and reports them next to the corrected tests (Fisher's exact test, permutation tests and the Benjamini–Hochberg correction; see its docstring).

## Command-line options

### `analyze.py`

| Option | Default | Meaning |
|---|---|---|
| `--L` | `2 3 4 5 6 7 8` | Synaptic depths |
| `--qber` | `0.01 … 0.11` | QBER values $p$ |
| `--K` | `3` | Hidden units |
| `--N` | `80` | Inputs per hidden unit |
| `--out` | `mapping_analysis.csv` | Output file |

### `select_mappings.py`

| Option | Default | Meaning |
|---|---|---|
| `--L` | `2 3 4 5 6 7 8` | Synaptic depths |
| `--max-all` | `20` | Simulate all placements if at most this many remain after mirror removal |
| `--out` | `selected_mappings.csv` | Output file |

### `simulate.py`

| Option | Default | Meaning |
|---|---|---|
| `--mappings` | `selected_mappings.csv` | Mappings to simulate (output of `select_mappings.py`) |
| `--L` | all | Only these synaptic depths |
| `--family` | all | Only these families (`modulo`, `hamming`, `placement`, `asym`, `clip`) |
| `--qber` | `0.01 … 0.11` | QBER values |
| `--runs` | `5000` | Runs per configuration |
| `--batch` | all runs | Runs simulated at once; lower to save memory |
| `--cap` | `1000000` | Iteration limit per run; runs that reach it are excluded and counted |
| `--K` | `3` | Hidden units |
| `--N` | `80` | Inputs per hidden unit |
| `--seed` | `20261115` | Base random seed |
| `--workers` | number of CPUs | Parallel processes |
| `--out` | `simulation_summary.csv` | Summary output file; configurations already in it are skipped |
| `--runs-out` | none | Optional gzipped CSV with the result of every single run |

### `evaluate.py`

| Option | Default | Meaning |
|---|---|---|
| `--analysis` | `mapping_analysis.csv` | Exact mapping properties (output of `analyze.py`) |
| `--summary` | `simulation_summary.csv` | One or more simulation summaries; the first occurrence of a configuration is used |
| `--family` | all | Only these families |
| `--out` | `evaluation_report.txt` | Report file; the tests against the baselines are written to `comparisons_with_modulo.csv` and `comparisons_with_clip.csv` |

## Mappings

Every mapping reads a block of $b$ bits as an unsigned integer $u$, most significant bit first, and assigns it a weight $f(u)$. The `family` column of all output files uses these names:

| `family` | Mapping | Weight range |
|---|---|---|
| `modulo` | Baseline: $f(u) = (u \bmod (2L+1)) - L$ | $\{-L, \dots, L\}$ |
| `placement` | Order-preserving placement $f_D$: the weight values in $D$ receive two consecutive patterns; `D` lists them, separated by `;` | $\{-L, \dots, L\}$ |
| `hamming` | Hamming weight: $f(u) = \mathrm{wt}(u) - \lceil b/2 \rceil$ | $\{-L, \dots, L\}$ |
| `asym` | Asymmetric one-to-one mapping $f(u) = u - L$, only for $L = 2, 4, 8$ | $\{-L, \dots, L-1\}$ |
| `clip` | Clipping mapping $f(u) = g(u - 2^{b-1})$: surplus patterns clipped to $\pm L$; simulated for all $L$ except 3 and 7, where it equals the placement $D = \{-L\}$ | $\{-L, \dots, L\}$ |

The symmetric mappings use $b = \lceil \log_2(2L+1) \rceil$ bits per weight, and the asymmetric mapping uses $b = \log_2 L + 1$. The recommended mapping for each $L$ is the placement with the smallest average locality (`lambda_avg`); ties are broken by the smallest `lambda_sq`.

## Output files

### `mapping_analysis.csv`

One row per mapping, QBER and pair (Alice–Bob or one of the three Alice–Eve models).

| Column | Paper notation | Meaning |
|---|---|---|
| `L`, `b`, `KN`, `n` | $L$, $b$, $KN$, $n$ | Synaptic depth, bits per weight, number of weights, key length $n = bKN$ |
| `family`, `D` | | Mapping (see above) |
| `H` | $H_f$ | Entropy per weight (bits) |
| `H_min` | $H_{\infty,f}$ | Min-entropy per weight (bits) |
| `loss_per_weight`, `loss_total` | $b - H_f$, $KN(b - H_f)$ | Entropy lost in the mapping |
| `mean` | $\mu_f$ | Mean initial weight |
| `lambda_avg`, `lambda_sq` | $\bar\Lambda_f$, $\bar\Lambda_f^{(2)}$ | Average absolute and squared weight change caused by a single bit flip |
| `pair` | | `alice_bob`, `eve_intercept_resend`, `eve_optimal_individual` or `eve_uncorrelated` |
| `qber`, `error_rate` | $p$, $p$ or $p_E$ | QBER and the bit error rate of the pair |
| `q`, `delta1`, `delta2` | $q_f$, $\delta_f^{(1)}$, $\delta_f^{(2)}$ | Per weight: probability that the two weights differ, expected absolute difference (initial distance), expected squared difference |

Eve's error rate is $p_E = 1/2 - p$ for the intercept-resend attack, $p_E = 1/2 - \sqrt{p(1-p)}$ for the optimal individual attack, and $p_E = 1/2$ for the uncorrelated attacker.

### `selected_mappings.csv`

One row per selected mapping, with its properties, the clipping range `w_min`, `w_max`, the full mapping `f` ($f(0), \dots, f(2^b - 1)$, space-separated) and the `reason` it was selected (`always`, `all`, `lambda_avg=…`, `lambda_sq_min` or `lambda_sq_max`).

### `simulation_summary.csv`

One row per mapping, QBER and attacker (six attackers: simple or geometric attack × three levels of initial knowledge). The iteration and leakage columns are identical in all six rows of a configuration.

| Column | Meaning |
|---|---|
| `runs`, `runs_capped` | Runs that synchronized within `--cap` iterations, and runs that did not |
| `iter_mean`, `iter_var` | Mean and variance of the number of iterations |
| `attack`, `pair`, `error_rate` | Attack (`simple`, `geometric`), the attacker's initial knowledge and her bit error rate $p_E$ |
| `full_success_rate` | Fraction of runs in which the attacker's weights equal Alice's when Alice and Bob synchronize |
| `overlap_init_mean`, `overlap_final_mean` | Mean fraction of the attacker's weights equal to Alice's, at the start and at synchronization |
| `overlap_increase_mean`, `overlap_increase_sd` | Partial success increase (final minus initial overlap): mean and standard deviation over runs |
| `overlap_final_sd` | Standard deviation of the final overlap over runs |
| `leak_tau` | Alice's outputs: one bit per iteration |
| `leak_flag` | Bob's outputs given Alice's: $\sum_t P(T \ge t)\, h(\epsilon_t)$ |
| `leak_stop` | $H(T)$, entropy of the number of iterations (plug-in estimate with the Miller–Madow correction) |
| `leak_comm` | Communication leakage $\ell_f$, the sum of the three terms above |
| `sync_H_sum` | Sum over the $KN$ weight positions of the entropy of the synchronized weight at that position (Miller–Madow), bits per key; an upper bound on the entropy of the synchronized weights, not conditioned on the public transcript |
| `sync_H_pooled`, `sync_Hmin_pooled` | Entropy and min-entropy per weight of all synchronized weights, pooled over positions and runs |

The optional `--runs-out` file has one row per run, with the number of iterations and each attacker's initial overlap, final overlap and full success.

## Simulation model

- Alice's key is uniformly random. Bob's key flips each bit with probability $p$ (the QBER), and each attacker's key flips each of Alice's bits with probability $p_E$. The simple and the geometric attacker with the same level of knowledge start from the same key.
- All machines map their keys to weights with the same mapping and clip their weights to its range.
- In each iteration, a public random input vector in $\{-1, +1\}^{KN}$ is drawn and the outputs are exchanged; $\mathrm{sgn}(0) = -1$. If Alice's and Bob's outputs agree, both update with the Hebbian rule the hidden units whose output equals their overall output.
- The attackers only observe. They update whenever Alice and Bob do, using Alice's output in place of their own. Before updating, the geometric attacker flips the hidden unit with the smallest absolute local field if her output differs from Alice's.
- A run stops when Alice's and Bob's weights are identical. This stopping condition is not available in a real protocol; it is used to study the dynamics of synchronization.
- At synchronization, Alice's weights are recorded, and the entropy of the synchronized weights is estimated across the runs of a configuration.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{  to be defined }
```

## License

MIT License. See [LICENSE](LICENSE).
