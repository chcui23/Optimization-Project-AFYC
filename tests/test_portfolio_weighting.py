"""Run with: python3 -m unittest discover -s tests -v."""

import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from portfolio_weighting import (
    prepare_inputs, build_benchmark_returns, representative_weights,
    fit_tracking_weights, reconstruct_clusters, weight_one_month,
)


def fixture():
    """Use Person 2's actual selection functions on a small synthetic universe."""
    rng = np.random.default_rng(9)
    months = pd.period_range('2000-01', periods=9, freq='M')
    permnos = np.arange(10, 18)
    returns = rng.normal(0.01, 0.05, (len(months), len(permnos)))
    history = pd.DataFrame(returns, index=months, columns=permnos)
    history.index.name, history.columns.name = 'month', 'permno'
    history = history.stack().rename('mthret').reset_index()
    formations = pd.MultiIndex.from_product([months, permnos], names=['month', 'permno']).to_frame(index=False)
    formations['mthcap'] = np.tile(np.arange(1, 9) * 100.0, len(months))
    # Execute only the two pure function definitions, never notebook data cells.
    namespace = dict(np=np, pd=pd, fcluster=fcluster)
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'part2.ipynb').read_text())
    wanted = {'choose_cluster_representatives', 'select_representatives_for_k'}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.FunctionDef) and node.name in wanted:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), 'part2.ipynb', 'exec'), namespace)
    month = months[-2]
    window = history.loc[history['month'].between(month - 5, month)]
    matrix = window.pivot(index='month', columns='permno', values='mthret')
    corr = matrix.corr()
    distance = np.clip(1 - corr.to_numpy(), 0, None)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method='average')
    selections = []
    for k in (3, 5):
        _, selected = namespace['select_representatives_for_k'](
            tree, corr, pd.DataFrame(distance, index=corr.index, columns=corr.columns), k)
        selected['month'] = month
        selections.append(selected)
    return formations, history, pd.concat(selections, ignore_index=True), month


class WeightingTests(unittest.TestCase):
    def test_representative_mass_and_coverage(self):
        universe = pd.DataFrame({'mthcap': [60, 20, 10, 10]}, index=[1, 2, 3, 4])
        members = pd.DataFrame({'permno': [1, 2, 3], 'cluster': [1, 2, 2]})
        selected = pd.DataFrame({'cluster': [1, 2], 'representative_permno': [1, 3]})
        weights, coverage = representative_weights(universe, members, selected, 'value_weighted')
        np.testing.assert_allclose(weights, [2 / 3, 1 / 3])
        self.assertAlmostEqual(coverage, 0.9)
        weights, coverage = representative_weights(universe, members, selected, 'equal_weighted')
        np.testing.assert_allclose(weights, [1 / 3, 2 / 3])
        self.assertAlmostEqual(coverage, 0.75)

    def test_exact_replication_and_label_alignment(self):
        rng = np.random.default_rng(1)
        x = pd.DataFrame(rng.normal(0, 0.03, (60, 3)), columns=[30, 10, 20])
        y = x @ np.array([0.2, 0.3, 0.5])
        initial = pd.Series([1 / 3] * 3, index=[10, 20, 30])
        weights, _ = fit_tracking_weights(x, y, initial)
        np.testing.assert_allclose(weights, [0.2, 0.3, 0.5], atol=1e-5)

    def test_cap_and_infeasible_cap(self):
        x = pd.DataFrame(np.random.default_rng(2).normal(0, 0.03, (60, 3)))
        initial = pd.Series([0.8, 0.1, 0.1])
        weights, _ = fit_tracking_weights(x, x[0], initial, max_weight=0.4)
        self.assertLessEqual(weights.max(), 0.4 + 1e-8)
        self.assertAlmostEqual(weights.sum(), 1)
        with self.assertRaisesRegex(ValueError, 'Infeasible'):
            fit_tracking_weights(x, x[0], initial, max_weight=0.3)

    def test_tracking_error_distinct_from_mse(self):
        z = np.linspace(-0.03, 0.03, 60)
        x = pd.DataFrame({1: z, 2: 2 * z + 0.1})
        y = pd.Series(z + 0.05)
        initial = pd.Series({1: 0.5, 2: 0.5})
        te, _ = fit_tracking_weights(x, y, initial)
        mse, _ = fit_tracking_weights(x, y, initial, objective='mse')
        self.assertGreater(te[1], 0.999)
        self.assertGreater(mse[2], 0.4)

    def test_singular_100_asset_problem(self):
        x = pd.DataFrame(np.random.default_rng(3).normal(0, 0.04, (60, 100)))
        initial = pd.Series(np.full(100, 0.01))
        y = x.iloc[:, :10].mean(axis=1)
        weights, _ = fit_tracking_weights(x, y, initial)
        self.assertGreaterEqual(weights.min(), 0)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertLess((x @ weights - y).var(), (x @ initial - y).var())

    def test_benchmark_uses_lagged_caps_and_rejects_missing_returns(self):
        formations, history, _, _ = fixture()
        benchmark = build_benchmark_returns(formations, history, expected_n=8)
        month = benchmark.index[0]
        realized = history.loc[history['month'].eq(month), 'mthret'].to_numpy()
        expected = np.arange(1, 9) @ realized / 36
        self.assertAlmostEqual(benchmark.loc[month, 'value_weighted'], expected)
        changed = formations.copy()
        changed.loc[changed['month'].eq(month), 'mthcap'] = np.arange(8, 0, -1)
        self.assertAlmostEqual(build_benchmark_returns(changed, history, 8).loc[month, 'value_weighted'], expected)
        history = history.drop(history.loc[history['month'].eq(month)].index[0])
        bad = build_benchmark_returns(formations, history, 8).loc[month]
        self.assertTrue(np.isnan(bad['value_weighted']))
        self.assertTrue(np.isnan(bad['equal_weighted']))
        self.assertEqual(bad['n_missing_returns'], 1)

    def test_handoff_and_no_future_information(self):
        formations, history, selections, month = fixture()
        formations, history, selections = prepare_inputs(formations, history, selections)
        benchmarks = build_benchmark_returns(formations, history, 8)
        weights, diagnostics = weight_one_month(
            month, formations, history, selections, benchmarks, lookback=6, expected_n=8)
        self.assertEqual(len(weights), (3 + 5) * 2 * 2)
        self.assertEqual(len(diagnostics), 8)
        self.assertTrue(weights['holding_month'].eq(str(month + 1)).all())
        sums = weights.groupby(['k', 'benchmark', 'method'])['weight'].sum()
        np.testing.assert_allclose(sums, 1)
        comparison = diagnostics.pivot(index=['k', 'benchmark'], columns='method', values='in_sample_te_annualized')
        self.assertTrue((comparison['optimized'] <= comparison['representative'] + 1e-7).all())
        future = history.copy()
        future.loc[future['month'] > month, 'mthret'] = 9.0
        future_formations = formations.copy()
        future_formations.loc[future_formations['month'] > month, 'mthcap'] = 999999
        changed, _ = weight_one_month(
            month, future_formations, future, selections,
            build_benchmark_returns(future_formations, future, 8), lookback=6, expected_n=8)
        pd.testing.assert_frame_equal(weights, changed)

    def test_benchmark_keeps_return_after_stock_leaves_universe(self):
        formations = pd.DataFrame({
            'month': pd.PeriodIndex(['2000-01', '2000-01', '2000-02', '2000-02'], freq='M'),
            'permno': [1, 2, 2, 3],
            'mthcap': [3.0, 1.0, 5.0, 5.0],
        })
        history = pd.DataFrame({
            'month': pd.PeriodIndex(['2000-02'] * 3, freq='M'),
            'permno': [1, 2, 3],
            'mthret': [0.10, -0.10, 0.80],
        })
        benchmark = build_benchmark_returns(formations, history, expected_n=2)
        february = benchmark.loc[pd.Period('2000-02', freq='M')]
        self.assertAlmostEqual(february['value_weighted'], 0.05)
        self.assertAlmostEqual(february['equal_weighted'], 0.0)
        self.assertEqual(february['n_missing_returns'], 0)

    def test_bad_selection_and_incomplete_training_stop(self):
        formations, history, selections, month = fixture()
        benchmarks = build_benchmark_returns(formations, history, 8)
        bad = selections.copy()
        bad.loc[bad.index[0], 'representative_permno'] = -1
        with self.assertRaisesRegex(ValueError, 'does not match'):
            reconstruct_clusters(month, formations, history, bad, lookback=6, expected_n=8)
        benchmarks.loc[month - 2, 'equal_weighted'] = np.nan
        with self.assertRaisesRegex(ValueError, 'incomplete benchmark'):
            weight_one_month(month, formations, history, selections, benchmarks, lookback=6, expected_n=8)

    def test_explicit_incomplete_benchmark_policy(self):
        formations, history, selections, month = fixture()
        benchmarks = build_benchmark_returns(formations, history, 8)
        benchmarks.loc[month - 2, 'equal_weighted'] = np.nan
        weights, diagnostics = weight_one_month(
            month, formations, history, selections, benchmarks,
            lookback=6, expected_n=8, missing_benchmark='drop', min_training_months=5,
        )
        self.assertTrue(diagnostics['n_training_months'].eq(5).all())
        self.assertTrue(diagnostics['n_excluded_training_months'].eq(1).all())
        self.assertTrue(diagnostics['excluded_training_months'].eq(str(month - 2)).all())
        np.testing.assert_allclose(
            weights.groupby(['k', 'benchmark', 'method'])['weight'].sum(), 1,
        )
        with self.assertRaisesRegex(ValueError, 'require at least 6'):
            weight_one_month(
                month, formations, history, selections, benchmarks,
                lookback=6, expected_n=8, missing_benchmark='drop', min_training_months=6,
            )


if __name__ == '__main__':
    unittest.main()
