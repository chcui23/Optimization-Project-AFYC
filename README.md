# Optimization-Project-AFYC

Small-stock portfolios tracking monthly top-500 CRSP value-weighted and
equal-weighted benchmarks.

1. `data_extraction.ipynb` — Person 1: export formations and monthly history to
   `top500_data/` (requires WRDS access). These two raw CSVs are not tracked in Git.
2. `part2.ipynb` — Person 2: select cluster representatives for k = 25, 50, 100.
   The resulting `cluster_selected_stocks.csv` is included.
3. **[`part3.ipynb`](part3.ipynb)** — Person 3: explanation, representative
   weighting, constrained tracking-error optimization, and exports for Person 4.
   Reusable functions are in [`portfolio_weighting.py`](portfolio_weighting.py).
4. **[`part4.ipynb`](part4.ipynb)** — Person 4: rolling out-of-sample evaluation,
   drift-adjusted turnover, common-date comparison, result tables, and charts.

For Part 3, use Python 3.11 or newer and install the dependencies in a virtual
environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Run the notebook from this directory after obtaining Person 1's exports.
Person 1's extraction also requires the `wrds` package and WRDS credentials.
The notebook and command-line exporter use the same monthly export function and
write matching weights, diagnostics, and benchmark audit files in `part3_output/`.
The notebook uses the `drop` policy described below, with at least 36 complete
observations in each 60-month training window.

Weights formed at month-end t apply to month t+1. Representative weights
normalize over the complete-history subset clustered by Person 2; optimization
targets the full 500-stock benchmark. The notebook explains this coverage gap,
missing-return checks, the missing January 1995 benchmark observation caused by
unavailable December 1994 holdings, and fitted versus out-of-sample results.
Person 4 should evaluate all strategies on the same holding dates and disclose
excluded training observations and unresolved realized returns.

Person 1's updated history export retains all available months for stocks ever
selected, including months after they leave the top 500. Part 3 uses that entire
history; it does not join it to same-month universe membership. The extraction
update in `0db0a4a` adds comments and `.copy()` without changing the row selection,
so it alone does not repair missing returns or imply different portfolio weights.

To export monthly stock weights for January 2000 through December 2025:

```sh
python export_monthly_weights.py --data-dir top500_data --start 2000-01 --end 2025-12 --missing-benchmark drop --min-training-months 36
```

Use `--data-dir` to point to Person 1's exports. The last holding month defaults
to the latest supported by the data (December 2025 in the current sample);
`--end YYYY-MM` can restrict it. January 2000 uses December 1999 formation weights.
The export contains both benchmarks, both weighting methods, and k = 25, 50, 100.
`holding_month`, `formation_month`, `k`, `benchmark`, `method`, `permno`, and
`weight` identify each position; weights are decimals and sum to one per portfolio.

`--missing-benchmark drop` uses only complete benchmark observations within each
60-month training window. The full-period command explicitly sets the minimum to
36 because earlier windows fall below the prior 48-observation threshold. It does
not fill missing returns or change Person 2's selections. The command-line defaults
remain `raise` and a minimum of 48; the flags above specify the extended-run policy.
Excluded dates and actual training counts are recorded in
`part3_output/monthly_weight_diagnostics_2000-01_to_2025-12.csv`, alongside
`monthly_weights_2000-01_to_2025-12.csv` and a benchmark coverage audit. Missing
realized benchmark returns remain missing for Person 4 to resolve.

Use the `2000-01_to_2025-12` files for the complete study. The older
`2020-01_to_2025-12` files remain available as the previous, shorter-period export.

Run the synthetic regression checks (no WRDS connection needed):

```sh
python3 -m unittest discover -s tests -v
```

Validated with Python 3.13, NumPy 2.3.5, pandas 2.3.3, and SciPy 1.16.3.
Regression tests and a complete notebook run on synthetic 500-stock data pass;
the real CRSP monthly export also passes validation for January 2000 through
December 2025 using the explicit incomplete-benchmark policy above (36–60 training
observations). It contains 218,400 stock-weight rows across 3,744 portfolios in
312 consecutive months. The 2020–2025 weights match the previous export exactly.
The monthly weights, diagnostics, and benchmark audit CSVs are in
`part3_output/`. Raw data, temporary results, and local
environments are excluded from Git.

## Part 4: backtesting

Open `part4.ipynb` from this directory and run its cells in order. It consumes
Person 1's two `top500_data/` CSVs (obtain these raw files from the team) and
Person 3's committed exports in `part3_output/`:

- `monthly_weights_2020-01_to_2025-12.csv`
- `monthly_weight_diagnostics_2020-01_to_2025-12.csv`
- `benchmark_returns_2020-01_to_2025-12.csv`

The current Part 4 holding period is **January 2020–December 2025**. Formation weights
apply only to the next month. Part 4 verifies Part 3's recorded policy of retaining
at least 48 complete observations within each 60-month training window, including
the exact excluded dates. It does not execute or modify Part 3 or refit weights.
Earlier exports named `portfolio_weights.csv` and `excluded_formation_months.csv`
are not used.

Part 4 reconstructs and verifies the benchmark export, validates portfolio keys
and weights, and compares all 12 strategies on the same valid dates. Missing
held-stock returns invalidate observations; zero-weight candidates do not require
a return. The coverage audit includes missing weight months and the reason each
strategy-month was omitted. Resolve upstream return/delisting issues before
drawing conclusions from an incomplete sample.

The saved real-data run evaluates all 12 strategies on **70 of 72 holding months**.
April 2020 and October 2025 are excluded because realized benchmark returns are
missing. Section 8 summarizes the results and limitations; refresh this narrative
if the inputs or experiment change.

Turnover compares new weights with the previous month's return-drifted holdings,
including stocks entering and exiting. Initial investment and undefined turnover
after gaps are excluded from the mean and counted explicitly. Cumulative returns
and CAGR restart in each continuous common block. Cumulative charts show the
longest block; the tables cover every block. All performance is gross of costs.

Outputs in ignored `part4_output/` include:

- `summary_statistics.csv`: tracking error, tracking difference, RMSE,
  correlation, and turnover with observation counts for all 12 strategies.
- `common_monthly_results.csv` and `continuous_block_statistics.csv`: matched
  returns, turnover, block wealth, cumulative returns, and CAGR.
- `all_strategy_months.csv`, `coverage_audit.csv`, `coverage_by_month.csv`,
  `invalid_held_stock_returns.csv`, and `missing_benchmark_constituents.csv`:
  realized-return coverage diagnostics.
- `training_coverage.csv`, `excluded_training_observations.csv`, and
  `upstream_weight_diagnostics.csv`: Person 3's training audit.
- `tracking_error_comparison.csv`, four PNG figures, upstream audit copies,
  and `run_manifest.json` with settings, versions, and input file hashes.

`tests/test_part4.py` exercises the notebook's functions directly and executes all
notebook code cells against a temporary synthetic 500-stock handoff. These tests
need no WRDS account and never write synthetic data into the real data folders.
Run them with the same unittest command above. Synthetic validation is not an
empirical backtest result.
