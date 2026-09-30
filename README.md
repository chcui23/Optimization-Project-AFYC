# Optimization-Project-AFYC

Small-stock portfolios tracking monthly top-500 CRSP value-weighted and
equal-weighted benchmarks.

1. `data_extraction.ipynb` — Person 1: export formations and monthly history to
   `top500_data/` (requires WRDS access). These two CSVs are not in this checkout.
2. `part2.ipynb` — Person 2: select cluster representatives for k = 25, 50, 100.
   The resulting `cluster_selected_stocks.csv` is included.
3. **[`part3.ipynb`](part3.ipynb)** — Person 3: explanation, representative
   weighting, constrained tracking-error optimization, and exports for Person 4.
   Reusable functions are in [`portfolio_weighting.py`](portfolio_weighting.py).

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
Part 3 writes
`part3_output/portfolio_weights.csv`, `weight_diagnostics.csv`,
`benchmark_returns.csv`, and `excluded_formation_months.csv`.

Weights formed at month-end t apply to month t+1. Representative weights
normalize over the complete-history subset clustered by Person 2; optimization
targets the full 500-stock benchmark. The notebook explains this coverage gap,
missing-return checks, the one-month startup delay caused by unavailable December
1994 holdings, and the distinction between fitted and out-of-sample results.
Person 4 should evaluate all strategies on the same dates and disclose or resolve
excluded formation months.

Run the synthetic regression checks (no WRDS connection needed):

```sh
python3 -m unittest discover -s tests -v
```

Validated with Python 3.13, NumPy 2.3.5, pandas 2.3.3, and SciPy 1.16.3.
Regression tests and a complete notebook run on synthetic 500-stock data pass;
the real CRSP run requires Person 1's exports. Generated data, results, and local
environments are excluded from Git.
