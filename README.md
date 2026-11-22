# execsim — market-agnostic execution simulator

An off-the-shelf tool for simulating and reporting VWAP-style execution
algorithms on any bar data (1-min, 5-min, any bar size; any market). No
dependency on this repo's data pipelines or any specific vendor schema.

## Quickstart

```python
import polars as pl
from execsim import simulate, simulate_many, summarize_all, write_report

# 1. bring your own bars: one row per bar
bars = pl.DataFrame(
    {
        "session": [...],  # session id (date or str), chronological
        "symbol": [...],
        "ts": [...],  # bar timestamp (orders bars within a session)
        "price": [...],  # bar traded VWAP (or best fill-price proxy)
        "volume": [...],  # bar traded volume
        # "spread_bps": [...]  # optional; default 2.0
    }
)

# 2. optional signal: return forecast for the NEXT bar, made at bar close
signal = pl.DataFrame(
    {
        "session": [...],
        "symbol": [...],
        "ts": [...],
        "p": [...],  # dimensionless return units (3.5e-4 == 3.5 bp)
    }
)

# 3. simulate -- one call per strategy, or several at once
res = simulate_many(
    bars,
    {
        "uniform": {"schedule": "uniform"},
        "static_vwap": {"schedule": "static", "curve_lookback": 20},
        "inst_vwap": {
            "schedule": "inst",
            "curve_lookback": 30,
            "weekday_conditioned": True,
            "weekday_col": "session",
        },
        "inst_tilted": {
            "schedule": "inst",
            "curve_lookback": 30,
            "weekday_conditioned": True,
            "weekday_col": "session",
            "kappa": 0.02,
        },
    },
    tilt_signal=signal,
)

# 4. summarize + report
summary = {k: summarize_all(v) for k, v in res.items()}
write_report(summary["inst_tilted"], "EXEC_REPORT.md")
```

See `examples/hk_1min.py` for a complete, runnable example on real HK 1-min
data (it reproduces the study in
the accompanying empirical study (not included in this repo)).

## Strategies

| schedule | what it does |
|---|---|
| `uniform` | equal slice per bar |
| `static`  | trailing volume-curve schedule (`curve_lookback` sessions, strictly earlier only) |
| `inst`    | institutional VWAP: weekday-conditioned curve (`weekday_col`, e.g. the session column if sessions are dates), volume-clock re-sync (`V̂ = EMA(realized_cum / curve_cum)`), urgency pacing (passive by default, cross the shortfall vs the previous bar's target), passive/aggressive spread accounting |

**Signal tilt** (requires `schedule="inst"`): each bar's slice is multiplied by
`exp(κ · sgn · p_prev · 1e4)`, clipped to 0.5–2× the curve share. `p_prev` is
the signal's value at the *previous* bar — the simulator applies the lag, so
the caller supplies forecasts as-is. `κ` is the volume elasticity per bp of
predicted move (0.01 = ±1%/bp; 0.02 recommended in the HK study). No
thresholding: all bars are adjusted proportionally, bounded by the clip.

## Causality (built in, audited)

- curves use **strictly earlier** sessions only;
- bar *b*'s schedule consumes realized volume **through bar b−1** and the
  signal of bar **b−1** (a forecast for bar b made at its open);
- the participation cap uses *predicted* volume `V̂·h[b]`, never bar b's
  realized volume;
- benchmarks (day VWAP, arrival = first bar price) are evaluation-only, never
  inputs to scheduling.

## Metrics

- `IS_day`: shortfall vs the session's VWAP, raw prices — ex-post benchmark,
  drift-contaminated; **compare strategies, not levels**.
- `IS_arr`: all-in shortfall vs arrival price (first bar's price), including
  spread accounting (market fills pay ½ spread; passive fills earn
  `passive_credit` × ½ spread; default credit 0.4).
- `completion` (filled/order; ≥0.99 required to count), `agg_frac` (share of
  size crossed aggressively). Day-clustered t-stats in `summarize_all`.

## Files

- `execsim/core.py` — everything (one importable module, stdlib + numpy + polars)
- `examples/hk_1min.py` — runnable example on real data
- `outputs/exec_example_report.md` — example output

## Background

Design, bug history, decomposition analyses, and the full HK study this
package distills: the accompanying empirical study (not included in this repo).

## Development

```bash
pip install -e . pytest ruff
ruff check .          # lint
ruff format .         # format
pytest tests/ -q      # tests (synthetic data, no network needed)
```

- **pre-commit**: `.pre-commit-config.yaml` runs ruff lint + format on every
  commit (`pre-commit install` once after cloning).
- **CI**: `.github/workflows/ci.yml` — ruff check + format-check, and pytest
  on Python 3.10–3.14, on every push and PR.
- **Python support**: 3.10–3.14 (`requires-python >=3.10`).
