"""Export each selected stock's weight for a continuous range of holding months."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_weighting import (
    build_benchmark_returns,
    prepare_inputs,
    weight_one_month,
)


def export_monthly_weights(
    formations, history, selections, output_dir='part3_output',
    start='2020-01', end=None, missing_benchmark='raise', min_training_months=48,
    objective='tracking_error', max_weight=1.0, ridge=0.0,
):
    """Shared notebook/CLI export; return weights, diagnostics, and benchmarks.

    Inputs are normalized with prepare_inputs. Keep the complete stock history:
    a stock's return remains relevant after it leaves the monthly top 500.
    Missing benchmark observations can be excluded explicitly, but requested
    holding months are never silently skipped.
    """
    formations, history, selections = prepare_inputs(formations, history, selections)
    benchmarks = build_benchmark_returns(formations, history)
    latest = min(
        history['month'].max(),
        formations['month'].max() + 1,
        selections['month'].max() + 1,
    )
    start = pd.Period(start, freq='M')
    end = pd.Period(end, freq='M') if end is not None else latest
    if start > end or end > latest:
        raise ValueError(
            f'Invalid holding range {start} to {end}; latest available is {latest}'
        )
    holding_months = pd.period_range(start, end, freq='M')
    k_values = {25, 50, 100}
    selections = selections.loc[selections['k'].isin(k_values)]

    all_weights, all_diagnostics = [], []
    for i, holding_month in enumerate(holding_months):
        formation_month = holding_month - 1
        selected = selections.loc[selections['month'].eq(formation_month)]
        if set(selected['k']) != k_values:
            raise ValueError(f'{formation_month}: missing selections for k=25, 50, or 100')
        weights, diagnostics = weight_one_month(
            formation_month, formations, history, selections, benchmarks,
            missing_benchmark=missing_benchmark,
            min_training_months=min_training_months,
            objective=objective, max_weight=max_weight, ridge=ridge,
        )
        all_weights.append(weights)
        all_diagnostics.append(diagnostics)
        if (i + 1) % 12 == 0 or i + 1 == len(holding_months):
            print(f'Completed {i + 1}/{len(holding_months)} holding months', flush=True)

    keys = ['holding_month', 'k', 'benchmark', 'method']
    weights = pd.concat(all_weights, ignore_index=True).sort_values(keys + ['permno'])
    diagnostics = pd.concat(all_diagnostics, ignore_index=True).sort_values(keys)
    checks = weights.groupby(keys)['weight'].agg(['sum', 'min', 'size'])
    if (
        len(checks) != len(holding_months) * 12
        or not np.allclose(checks['sum'], 1, atol=1e-7, rtol=0)
        or not checks['min'].ge(0).all()
        or not np.array_equal(checks['size'], checks.index.get_level_values('k'))
        or weights.duplicated(keys + ['permno']).any()
        or not np.isfinite(weights['weight']).all()
    ):
        raise RuntimeError('Monthly weight validation failed; no CSVs written')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'{start}_to_{end}.csv'
    weight_path = output_dir / f'monthly_weights_{suffix}'
    weights.to_csv(weight_path, index=False)
    diagnostics.to_csv(output_dir / f'monthly_weight_diagnostics_{suffix}', index=False)
    # Keep the target coverage audit alongside the weights for the backtester.
    benchmarks.loc[start - 60:end].to_csv(output_dir / f'benchmark_returns_{suffix}')
    print(f'Saved {len(weights):,} stock-weight rows to {weight_path}')
    print(
        'Complete training months per formation:',
        diagnostics['n_training_months'].min(), 'to',
        diagnostics['n_training_months'].max(),
    )
    return weights, diagnostics, benchmarks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('top500_data'))
    parser.add_argument('--selections', type=Path, default=Path('cluster_selected_stocks.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('part3_output'))
    parser.add_argument('--start', default='2020-01', help='First holding month (YYYY-MM)')
    parser.add_argument('--end', help='Last holding month; defaults to available data end')
    parser.add_argument('--missing-benchmark', choices=['raise', 'drop'], default='raise')
    parser.add_argument('--min-training-months', type=int, default=48)
    args = parser.parse_args()

    paths = [
        args.data_dir / 'top500_formations.csv',
        args.data_dir / 'selected_stock_monthly_history.csv',
        args.selections,
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error('Missing input files: ' + ', '.join(missing))
    export_monthly_weights(
        *(pd.read_csv(path) for path in paths),
        output_dir=args.output_dir, start=args.start, end=args.end,
        missing_benchmark=args.missing_benchmark,
        min_training_months=args.min_training_months,
    )


if __name__ == '__main__':
    main()
