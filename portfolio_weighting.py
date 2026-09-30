"""Person 3: portfolio weights using Person 1/2's exported data.

All returns are decimal monthly total returns. A formation at month t uses
information through t and produces weights for holding month t+1.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


BENCHMARKS = ("value_weighted", "equal_weighted")


def prepare_inputs(formations, history, selections):
    """Copy, normalize dates, and check the three existing CSV schemas."""
    specs = [
        (formations, ['month', 'permno', 'mthcap'], ['month', 'permno']),
        (history, ['month', 'permno', 'mthret'], ['month', 'permno']),
        (
            selections,
            ['month', 'k', 'cluster', 'representative_permno', 'cluster_size'],
            ['month', 'k', 'cluster'],
        ),
    ]
    result = []
    for frame, required, keys in specs:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')
        frame = frame.copy()
        frame['month'] = pd.PeriodIndex(frame['month'], freq='M')
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f'Null or duplicate keys: {keys}')
        result.append(frame)
    return tuple(result)


def build_benchmark_returns(formations, history, expected_n=500):
    """Full-universe benchmark returns, using prior month-end holdings.

    Missing constituent returns invalidate the entire benchmark month. They
    are never filled with zero or dropped and reweighted. Return coverage is
    exported so Person 1 can resolve any missing/delisting observations.
    """
    last_month = history['month'].max()
    holdings = formations.loc[
        formations['month'] + 1 <= last_month, ['month', 'permno', 'mthcap']
    ].copy()
    holdings['month'] = holdings['month'] + 1
    joined = holdings.merge(
        history[['month', 'permno', 'mthret']],
        on=['month', 'permno'],
        how='left',
        validate='one_to_one',
    )
    rows = []
    for month, group in joined.groupby('month', sort=True):
        caps = group['mthcap'].to_numpy(dtype=float)
        returns = group['mthret'].to_numpy(dtype=float)
        if len(group) != expected_n or not np.isfinite(caps).all() or (caps <= 0).any():
            raise ValueError(f'{month}: invalid benchmark universe/capitalizations')
        valid = np.isfinite(returns) & (returns >= -1)
        complete = valid.all()
        rows.append({
            'month': month,
            'formation_month': month - 1,
            'n_stocks': len(group),
            'n_missing_returns': int((~valid).sum()),
            'value_weighted': float(caps @ returns / caps.sum()) if complete else np.nan,
            'equal_weighted': float(returns.mean()) if complete else np.nan,
        })
    if not rows:
        raise ValueError('No holding months overlap the exported history')
    return pd.DataFrame(rows).set_index('month')


def reconstruct_clusters(
    formation_month, formations, history, selections, lookback=60, expected_n=500
):
    """Reproduce Part 2's average-linkage clustering and check its saved CSV.

    Membership is needed to sum capitalization within each cluster; cluster
    size and the representative's own capitalization alone are insufficient.
    """
    month = pd.Period(formation_month, freq='M')
    universe = formations.loc[formations['month'].eq(month)].set_index('permno')
    if len(universe) != expected_n or not universe.index.is_unique:
        raise ValueError(f'{month}: expected {expected_n} unique universe stocks')
    caps = universe['mthcap'].to_numpy(dtype=float)
    if not np.isfinite(caps).all() or (caps <= 0).any():
        raise ValueError(f'{month}: invalid market capitalization')
    calendar = pd.period_range(month - lookback + 1, month, freq='M')
    window = history.loc[
        history['month'].isin(calendar) & history['permno'].isin(universe.index)
    ]
    returns = window.pivot(index='month', columns='permno', values='mthret')
    returns = returns.reindex(index=calendar, columns=sorted(universe.index))
    returns = returns.loc[:, returns.notna().all()]
    values = returns.to_numpy(dtype=float)
    if returns.shape[1] < 2 or not np.isfinite(values).all() or (values < -1).any():
        raise ValueError(f'{month}: invalid eligible return history')
    corr = returns.corr().to_numpy()
    if not np.isfinite(corr).all():
        raise ValueError(f'{month}: undefined correlations (check constant returns)')
    distance = np.clip(1 - corr, 0, None)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method='average')
    saved = selections.loc[selections['month'].eq(month)]
    if saved.empty:
        raise ValueError(f'{month}: no saved representatives')
    memberships = []
    for k, selected in saved.groupby('k', sort=True):
        labels = fcluster(tree, t=int(k), criterion='maxclust')
        members = pd.DataFrame({'permno': returns.columns, 'cluster': labels})
        if len(selected) != k or members['cluster'].nunique() != k:
            raise ValueError(f'{month}, k={k}: expected exactly k clusters')
        reconstructed = []
        for cluster, group in members.groupby('cluster'):
            positions = returns.columns.get_indexer(group['permno'])
            mean_distances = distance[np.ix_(positions, positions)].mean(axis=1)
            medoid = group['permno'].iloc[np.argmin(mean_distances)]
            reconstructed.append((cluster, medoid, len(group)))
        actual = pd.DataFrame(
            reconstructed,
            columns=['cluster', 'representative_permno', 'cluster_size'],
        ).sort_values('cluster')
        expected = selected[actual.columns].sort_values('cluster')
        if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
            raise ValueError(
                f'{month}, k={k}: clustering does not match Part 2 CSV; '
                'use the original data/SciPy environment or export '
                'memberships from Part 2'
            )
        members['k'] = int(k)
        memberships.append(members)
    return universe, returns, pd.concat(memberships, ignore_index=True)


def representative_weights(universe, members, selected, benchmark):
    """Sum eligible benchmark mass by cluster, then normalize to invest 100%.

    Coverage is measured against the full universe before normalization.
    Excluded stocks have no cluster in Part 2 and cannot be represented here.
    """
    if benchmark == 'value_weighted':
        base = universe['mthcap'] / universe['mthcap'].sum()
    elif benchmark == 'equal_weighted':
        base = pd.Series(1 / len(universe), index=universe.index)
    else:
        raise ValueError('Unknown benchmark')
    members = members.copy()
    members['mass'] = members['permno'].map(base)
    if members['mass'].isna().any() or members['permno'].duplicated().any():
        raise ValueError('Invalid cluster membership')
    masses = members.groupby('cluster')['mass'].sum()
    selected = selected.sort_values('cluster')
    weights = selected['cluster'].map(masses).to_numpy(dtype=float)
    coverage = float(masses.sum())
    if (
        not np.isfinite(weights).all()
        or coverage <= 0
        or not np.isclose(weights.sum(), coverage)
    ):
        raise ValueError('Representatives do not cover all eligible clusters')
    weights = pd.Series(
        weights / coverage,
        index=selected['representative_permno'].to_numpy(),
        name='weight',
    )
    return weights, coverage


def fit_tracking_weights(
    returns, benchmark_returns, initial_weights,
    objective='tracking_error', max_weight=1.0, ridge=0.0,
):
    """Constrained quadratic fit, without covariance inversion.

    tracking_error minimizes sample variance of active returns; mse also
    penalizes their mean. Optional ridge shrinks toward representative weights.
    The default ridge=0 is the exact unregularized assignment objective.
    """
    if objective not in ('tracking_error', 'mse'):
        raise ValueError('objective must be tracking_error or mse')
    if not returns.index.equals(benchmark_returns.index):
        raise ValueError('Stock and benchmark months must match exactly')
    if (
        not returns.columns.is_unique
        or not returns.index.is_unique
        or not initial_weights.index.is_unique
    ):
        raise ValueError('Duplicate stock or month labels')
    target = initial_weights.reindex(returns.columns).to_numpy(dtype=float)
    x = returns.to_numpy(dtype=float)
    y = benchmark_returns.to_numpy(dtype=float)
    if len(y) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Need at least two complete, finite training months')
    if (
        not np.isfinite(target).all()
        or (target < 0).any()
        or not np.isclose(target.sum(), 1)
    ):
        raise ValueError('Initial weights must be long-only and sum to one')
    k = x.shape[1]
    if (
        not np.isfinite(max_weight)
        or not 0 < max_weight <= 1
        or k * max_weight < 1 - 1e-12
    ):
        raise ValueError('Infeasible maximum weight: require k * max_weight >= 1')
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError('ridge must be finite and nonnegative')
    if objective == 'tracking_error':
        x = x - x.mean(axis=0)
        y = y - y.mean()
        denominator = len(y) - 1
    else:
        denominator = len(y)
    # Scale the entire objective to help SLSQP handle decimal monthly returns.
    scale = max(float(np.mean(x * x)), float(np.mean(y * y)), ridge, 1e-8)

    def loss(w):
        error = x @ w - y
        penalty = ridge * np.sum((w - target) ** 2)
        return (error @ error / denominator + penalty) / scale

    def gradient(w):
        tracking_gradient = 2 * x.T @ (x @ w - y) / denominator
        return (tracking_gradient + 2 * ridge * (w - target)) / scale

    start = target.copy() if target.max() <= max_weight else np.full(k, 1 / k)
    result = minimize(
        loss,
        start,
        jac=gradient,
        method='SLSQP',
        bounds=[(0, max_weight)] * k,
        constraints=[{
            'type': 'eq',
            'fun': lambda w: w.sum() - 1,
            'jac': lambda w: np.ones(k),
        }],
        options={'ftol': 1e-11, 'maxiter': 2000},
    )
    w = result.x
    if (
        not result.success
        or not np.isfinite(w).all()
        or abs(w.sum() - 1) > 1e-7
        or w.min() < -1e-8
        or w.max() > max_weight + 1e-8
        or loss(w) > loss(start) + 1e-8
    ):
        raise RuntimeError(f'Weight optimization failed: {result.message}')
    # Remove only floating-point bound noise, then recheck feasibility.
    w = np.clip(w, 0, max_weight)
    w /= w.sum()
    if w.max() > max_weight + 1e-8:
        raise RuntimeError('Maximum weight violated after numerical cleanup')
    return pd.Series(w, index=returns.columns, name='weight'), {
        'solver_message': str(result.message),
        'solver_iterations': int(result.nit),
    }


def _tracking_diagnostics(stock_returns, benchmark_returns, weights):
    """Calculate fitted tracking metrics for one portfolio."""
    active = stock_returns @ weights - benchmark_returns
    return {
        'n_positive_weights': int((weights > 1e-8).sum()),
        'max_weight': float(weights.max()),
        'in_sample_te_annualized': float(active.std(ddof=1) * np.sqrt(12)),
        'in_sample_mean_active_monthly': float(active.mean()),
        'in_sample_rmse_monthly': float(np.sqrt(np.mean(active ** 2))),
    }


def weight_one_month(
    formation_month, formations, history, selections, benchmarks,
    lookback=60, expected_n=500, objective='tracking_error',
    max_weight=1.0, ridge=0.0, missing_benchmark='raise', min_training_months=48,
):
    """Return long-format weights and in-sample diagnostics for both benchmarks.

    By default an incomplete benchmark window fails. With missing_benchmark
    set to 'drop', fit only complete benchmark months within the same lookback
    window, require min_training_months, and record all exclusions. Stock
    selection still requires the full history. No t+1 returns are used.
    """
    month = pd.Period(formation_month, freq='M')
    calendar = pd.period_range(month - lookback + 1, month, freq='M')
    training = benchmarks.reindex(calendar)[list(BENCHMARKS)]
    complete_months = np.isfinite(training.to_numpy(dtype=float)).all(axis=1)
    bad = training.index[~complete_months].astype(str).tolist()
    if missing_benchmark not in ('raise', 'drop'):
        raise ValueError("missing_benchmark must be 'raise' or 'drop'")
    if missing_benchmark == 'raise' and bad:
        raise ValueError(f'{month}: incomplete benchmark training window: {bad}')
    if missing_benchmark == 'drop':
        if not 2 <= min_training_months <= lookback:
            raise ValueError('Require 2 <= min_training_months <= lookback')
        training = training.loc[complete_months]
        if len(training) < min_training_months:
            raise ValueError(
                f'{month}: only {len(training)} complete benchmark months; '
                f'require at least {min_training_months}'
            )
    universe, returns, memberships = reconstruct_clusters(
        month, formations, history, selections, lookback, expected_n
    )
    weight_rows, diagnostic_rows = [], []
    monthly_selections = selections.loc[selections['month'].eq(month)]
    for k, members in memberships.groupby('k', sort=True):
        selected = monthly_selections.loc[monthly_selections['k'].eq(k)]
        for benchmark in BENCHMARKS:
            representative, coverage = representative_weights(
                universe, members, selected, benchmark
            )
            x = returns.loc[training.index, representative.index]
            y = training[benchmark]
            optimized, solver = fit_tracking_weights(
                x, y, representative,
                objective=objective, max_weight=max_weight, ridge=ridge,
            )
            methods = {'representative': representative, 'optimized': optimized}
            for method, weights in methods.items():
                is_optimized = method == 'optimized'
                keys = {
                    'formation_month': str(month),
                    'holding_month': str(month + 1),
                    'k': int(k),
                    'benchmark': benchmark,
                    'method': method,
                }
                for permno, weight in weights.items():
                    weight_rows.append({
                        **keys, 'permno': int(permno), 'weight': float(weight),
                    })
                diagnostic_rows.append({
                    **keys,
                    'training_start': str(calendar[0]),
                    'training_end': str(month),
                    'n_training_months': len(training),
                    'missing_benchmark_policy': missing_benchmark,
                    'n_excluded_training_months': len(bad),
                    'excluded_training_months': ';'.join(bad),
                    'n_eligible': returns.shape[1],
                    'benchmark_mass_covered': coverage,
                    **_tracking_diagnostics(x, y, weights),
                    'objective': objective if is_optimized else 'cluster_mass',
                    'ridge': ridge if is_optimized else 0.0,
                    'weight_cap': max_weight if is_optimized else 1.0,
                    'solver_message': (
                        solver['solver_message'] if is_optimized else 'not applicable'
                    ),
                })
    return pd.DataFrame(weight_rows), pd.DataFrame(diagnostic_rows)
