"""Backtest regression checks; no WRDS connection or real return files required."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NOTEBOOK = Path(__file__).resolve().parents[1] / "part4.ipynb"
notebook = json.loads(NOTEBOOK.read_text())
namespace = dict(np=np, pd=pd, plt=plt,
                 K_VALUES=(25, 50, 100),
                 BENCHMARKS=("value_weighted", "equal_weighted"),
                 METHODS=("representative", "optimized"),
                 STRATEGY_KEYS=["k", "benchmark", "method"],
                 PORTFOLIO_KEYS=["holding_month", "k", "benchmark", "method"])
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
evaluate = namespace["evaluate_portfolios"]
common = namespace["common_sample"]
summarize = namespace["summarize"]
validate_training = namespace["validate_training_handoff"]


def fixture():
    weights, history, benchmarks = [], [], []
    for month, ids, values, returns in [
        ("2000-02", [1, 2], [0.5, 0.5], [0.2, 0.0, 0.5]),
        ("2000-03", [2, 3], [0.25, 0.75], [0.9, 0.04, -0.08]),
        ("2000-04", [1, 3], [0.3, 0.7], [0.1, -0.1, 0.02]),
    ]:
        for benchmark in namespace["BENCHMARKS"]:
            for method in namespace["METHODS"]:
                for permno, weight in zip(ids, values):
                    weights.append(dict(formation_month=str(pd.Period(month) - 1),
                                        holding_month=month, k=2, benchmark=benchmark,
                                        method=method, permno=permno, weight=weight))
        for permno, ret in zip([1, 2, 3], returns):
            history.append(dict(month=month, permno=permno, mthret=ret))
        benchmarks.append(dict(month=month, value_weighted=0.01, equal_weighted=0.02))
    return pd.DataFrame(weights), pd.DataFrame(history), pd.DataFrame(benchmarks)


class BacktestTests(unittest.TestCase):
    def test_next_month_alignment_and_drifted_turnover_including_exits(self):
        w, h, b = fixture()
        monthly, _ = evaluate(w.sample(frac=1, random_state=5), h.sample(frac=1), b, (2,))
        g = monthly.query("benchmark == 'value_weighted' and method == 'representative'").set_index("holding_month")
        self.assertAlmostEqual(g.loc[pd.Period("2000-02"), "portfolio_return"], 0.1)
        self.assertAlmostEqual(g.loc[pd.Period("2000-03"), "portfolio_return"], -0.05)
        # Old holdings drift to (6/11, 5/11, 0); target is (0, 1/4, 3/4).
        self.assertAlmostEqual(g.loc[pd.Period("2000-03"), "turnover"], 0.75)
        self.assertTrue(np.isnan(g.iloc[0]["turnover"]))
        self.assertEqual(g.iloc[1]["turnover_status"], "observed")

    def test_future_returns_do_not_change_earlier_observations_or_turnover(self):
        w, h, b = fixture()
        original, _ = evaluate(w, h, b, (2,))
        h.loc[h.month.eq("2000-04"), "mthret"] = 0.7
        changed, _ = evaluate(w, h, b, (2,))
        pd.testing.assert_frame_equal(original.iloc[:8], changed.iloc[:8])
        np.testing.assert_allclose(original.turnover, changed.turnover, equal_nan=True)

    def test_missing_returns_exclude_common_month_and_break_wealth(self):
        w, h, b = fixture()
        h.loc[h.month.eq("2000-03") & h.permno.eq(3), "mthret"] = np.nan
        monthly, _ = evaluate(w, h, b, (2,))
        panel, audit = common(monthly, "2000-02", "2000-04", (2,))
        self.assertEqual(len(panel), 8)
        self.assertEqual(panel.block.nunique(), 2)
        self.assertFalse(audit.loc[audit.holding_month.eq(pd.Period("2000-03")), "valid"].any())
        summary, blocks, curves = summarize(panel)
        self.assertTrue(summary.n_blocks.eq(2).all())
        april = curves.loc[curves.holding_month.eq(pd.Period("2000-04"))]
        np.testing.assert_allclose(april.portfolio_wealth, 1.044)
        self.assertTrue(april.turnover.isna().all())
        self.assertTrue(blocks.n_months.eq(1).all())

    def test_zero_weight_missing_return_is_unused(self):
        w, h, b = fixture()
        feb = w.holding_month.eq("2000-02")
        w.loc[feb & w.permno.eq(1), "weight"] = 1.0
        w.loc[feb & w.permno.eq(2), "weight"] = 0.0
        h.loc[h.month.eq("2000-02") & h.permno.eq(2), "mthret"] = np.nan
        monthly, _ = evaluate(w, h, b, (2,))
        self.assertTrue(monthly.valid.all())
        np.testing.assert_allclose(monthly.iloc[:4].portfolio_return, 0.2)
        np.testing.assert_allclose(monthly.iloc[4:8].turnover, 1.0)

    def test_invalid_handoff_rejected(self):
        w, h, b = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate(pd.concat([w, w.iloc[:1]]), h, b, (2,))
        bad = w.copy()
        bad.loc[0, "holding_month"] = "1999-01"
        with self.assertRaisesRegex(ValueError, "exactly one month"):
            evaluate(bad, h, b, (2,))
        bad = w.copy()
        bad.loc[0, "weight"] = -0.1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            evaluate(bad, h, b, (2,))
        bad = w.copy()
        bad.loc[0, "weight"] = 0.2
        with self.assertRaisesRegex(ValueError, "sum to one"):
            evaluate(bad, h, b, (2,))

    def test_missing_strategy_month_is_audited_not_silently_omitted(self):
        w, h, b = fixture()
        w = w.loc[~(w.holding_month.eq("2000-03") & w.method.eq("optimized"))]
        monthly, _ = evaluate(w, h, b, (2,))
        panel, audit = common(monthly, "2000-02", "2000-05", (2,))
        self.assertEqual(panel.holding_month.nunique(), 2)
        self.assertEqual(len(audit), 16)
        self.assertEqual(audit.reason.eq("missing Part 3 weights").sum(), 6)
        april_opt = monthly.loc[monthly.holding_month.eq(pd.Period("2000-04")) & monthly.method.eq("optimized")]
        self.assertTrue(april_opt.turnover.isna().all())

    def test_missing_benchmark_and_return_below_minus_one(self):
        w, h, b = fixture()
        b.loc[1, "equal_weighted"] = np.nan
        monthly, _ = evaluate(w, h, b, (2,))
        panel, _ = common(monthly, "2000-02", "2000-04", (2,))
        self.assertEqual(panel.holding_month.nunique(), 2)
        h.loc[h.month.eq("2000-02") & h.permno.eq(1), "mthret"] = -1.1
        monthly, _ = evaluate(w, h, b, (2,))
        self.assertTrue(monthly.iloc[:4].portfolio_return.isna().all())

    def test_metric_formulas(self):
        w, h, b = fixture()
        monthly, _ = evaluate(w, h, b, (2,))
        panel, _ = common(monthly, "2000-02", "2000-04", (2,))
        summary, blocks, _ = summarize(panel)
        row = summary.query("benchmark == 'value_weighted' and method == 'representative'").iloc[0]
        active = np.array([0.1, -0.05, 0.044]) - 0.01
        self.assertAlmostEqual(row.tracking_error_annualized, active.std(ddof=1) * np.sqrt(12))
        self.assertAlmostEqual(row.mean_tracking_difference_annualized, 12 * active.mean())
        self.assertAlmostEqual(row.rmse_monthly, np.sqrt(np.mean(active ** 2)))
        self.assertAlmostEqual(blocks.iloc[0].portfolio_cumulative_return, 1.1 * 0.95 * 1.044 - 1)

    def test_complete_loss_has_no_defined_next_turnover(self):
        w, h, b = fixture()
        h.loc[h.month.eq("2000-02"), "mthret"] = -1
        monthly, _ = evaluate(w, h, b, (2,))
        self.assertTrue(monthly.iloc[4:8].turnover.isna().all())
        self.assertTrue(monthly.iloc[4:8].turnover_status.eq("previous portfolio depleted").all())

    def test_notebook_end_to_end_on_synthetic_handoff(self):
        from portfolio_weighting import build_benchmark_returns
        # Synthetic data are temporary and never placed in the real data directories.
        with tempfile.TemporaryDirectory(prefix="part4-test-") as folder:
            root = Path(folder)
            (root / "top500_data").mkdir()
            (root / "part3_output").mkdir()
            months = pd.period_range("1995-01", "2000-03", freq="M")
            ids = np.arange(1, 501)
            f = pd.MultiIndex.from_product([months, ids], names=["month", "permno"]).to_frame(index=False)
            f["mthcap"] = 100.0
            h = f[["month", "permno"]].copy()
            h["mthret"] = np.repeat(0.01 + 0.02 * np.sin(np.arange(len(months))), 500)
            # Exercise Part 3's new drop policy: one incomplete TRAINING month.
            h = h.loc[~(h.month.eq(pd.Period("1998-03")) & h.permno.eq(500))]
            f.to_csv(root / "top500_data/top500_formations.csv", index=False)
            h.to_csv(root / "top500_data/selected_stock_monthly_history.csv", index=False)
            suffix = "2000-01_to_2025-12.csv"
            build_benchmark_returns(f, h).to_csv(root / f"part3_output/benchmark_returns_{suffix}")
            rows, diagnostics = [], []
            for month in pd.period_range("2000-01", "2000-03", freq="M"):
                for k in (25, 50, 100):
                    for benchmark in namespace["BENCHMARKS"]:
                        for method in namespace["METHODS"]:
                            keys = dict(holding_month=str(month), formation_month=str(month - 1),
                                        k=k, benchmark=benchmark, method=method)
                            first = month == pd.Period("2000-01")
                            diagnostics.append(dict(**keys, training_start=str(month - 60),
                                                    training_end=str(month - 1), n_training_months=58 if first else 59,
                                                    missing_benchmark_policy="drop",
                                                    n_excluded_training_months=2 if first else 1,
                                                    excluded_training_months="1995-01;1998-03" if first else "1998-03"))
                            rows.extend(dict(**keys, permno=int(i), weight=1/k) for i in ids[:k])
            pd.DataFrame(rows).to_csv(root / f"part3_output/monthly_weights_{suffix}", index=False)
            pd.DataFrame(diagnostics).to_csv(root / f"part3_output/monthly_weight_diagnostics_{suffix}", index=False)
            # Old exports are deliberately unusable: Part 4 must ignore them.
            pd.DataFrame({"stale": [True]}).to_csv(root / "part3_output/portfolio_weights.csv", index=False)
            previous = Path.cwd()
            try:
                os.chdir(root)
                ns = {"__name__": "__main__"}
                for idx, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), f"part4.ipynb cell {idx}", "exec"), ns)
            finally:
                os.chdir(previous)
                plt.close("all")
            results = pd.read_csv(root / "part4_output/summary_statistics.csv")
            self.assertEqual(len(results), 12)
            self.assertTrue(results.n_months.eq(3).all())
            np.testing.assert_allclose(results.tracking_error_annualized, 0, atol=1e-14)
            np.testing.assert_allclose(results.correlation, 1, atol=1e-12)
            self.assertEqual(len(list((root / "part4_output").glob("*.png"))), 4)
            manifest = json.loads((root / "part4_output/run_manifest.json").read_text())
            self.assertEqual(manifest["common_months"], 3)
            self.assertEqual(manifest["actual_training_observations_min"], 58)
            self.assertEqual(manifest["minimum_training_observations"], 36)
            self.assertEqual(manifest["requested_start"], "2000-01")
            self.assertEqual(manifest["training_policy"], ["drop"])
            self.assertTrue(pd.read_csv(root / "part4_output/missing_benchmark_constituents.csv").empty)

    def test_training_policy_counts_dates_and_minimum(self):
        w, _, _ = fixture()
        b = pd.DataFrame({"value_weighted": 0.01, "equal_weighted": 0.02},
                         index=pd.period_range("1999-08", "2000-04", freq="M"))
        b.loc[pd.Period("1999-12"), :] = np.nan
        rows = []
        for row in w.drop_duplicates(namespace["PORTFOLIO_KEYS"]).itertuples(index=False):
            form = pd.Period(row.formation_month)
            rows.append(dict(holding_month=row.holding_month, formation_month=str(form),
                             k=row.k, benchmark=row.benchmark, method=row.method,
                             training_start=str(form-5), training_end=str(form),
                             n_training_months=5, missing_benchmark_policy="drop",
                             n_excluded_training_months=1, excluded_training_months="1999-12"))
        d = pd.DataFrame(rows)
        checked, excluded = validate_training(d, w, b, lookback=6, min_training=4)
        self.assertEqual(len(excluded), 12)
        self.assertTrue(checked.n_training_months.eq(5).all())
        with self.assertRaisesRegex(ValueError, "fewer than"):
            validate_training(d, w, b, lookback=6, min_training=6)
        bad = d.copy()
        bad.loc[0, "excluded_training_months"] = "2000-01"
        with self.assertRaisesRegex(ValueError, "excluded training dates"):
            validate_training(bad, w, b, lookback=6, min_training=4)
        bad = d.copy()
        bad.loc[0, "n_training_months"] = 6
        with self.assertRaisesRegex(ValueError, "observation counts"):
            validate_training(bad, w, b, lookback=6, min_training=4)
        bad = d.copy()
        bad.loc[0, "missing_benchmark_policy"] = "raise"
        with self.assertRaisesRegex(ValueError, "raise policy"):
            validate_training(bad, w, b, lookback=6, min_training=4)

    def test_extended_training_threshold_accepts_36_and_rejects_35(self):
        w, _, _ = fixture()
        w = w.loc[w.holding_month.eq("2000-02")]
        calendar = pd.period_range("1995-02", "2000-01", freq="M")
        b = pd.DataFrame({"value_weighted": 0.01, "equal_weighted": 0.02}, index=calendar)
        b.iloc[:24] = np.nan
        d = w.drop_duplicates(namespace["PORTFOLIO_KEYS"])[namespace["PORTFOLIO_KEYS"]].copy()
        d["formation_month"] = "2000-01"
        d["training_start"], d["training_end"] = "1995-02", "2000-01"
        d["missing_benchmark_policy"] = "drop"
        d["n_training_months"], d["n_excluded_training_months"] = 36, 24
        d["excluded_training_months"] = ";".join(calendar[:24].astype(str))
        checked, _ = validate_training(d, w, b)
        self.assertTrue(checked.n_training_months.eq(36).all())
        b.iloc[24] = np.nan
        d["n_training_months"], d["n_excluded_training_months"] = 35, 25
        d["excluded_training_months"] = ";".join(calendar[:25].astype(str))
        with self.assertRaisesRegex(ValueError, "fewer than 36"):
            validate_training(d, w, b)


if __name__ == "__main__":
    unittest.main()
