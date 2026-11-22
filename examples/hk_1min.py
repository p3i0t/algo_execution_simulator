"""Example: reproduce the HK 1-min execution study with the execsim package.

Run from the repo root:
    python projects/execution_tool/examples/hk_1min.py
"""

import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from execsim import load_signal, simulate_many, summarize_all, write_report  # noqa: E402

SNAP = REPO / "data/hk/intraday/1min/snapshot"
PREDS = REPO / "projects/hk_algo_execution/outputs"


def load_bars(symbols: list[str], start: str, end: str) -> pl.DataFrame:
    files = sorted(str(p) for p in SNAP.glob("*.parquet") if start <= p.stem <= end)
    lf = pl.scan_parquet(files).filter(pl.col("symbol").is_in(symbols))
    t = pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32)
    lf = lf.filter(((t >= 9 * 60 + 30) & (t < 12 * 60)) | ((t >= 13 * 60) & (t < 16 * 60)))
    df = (
        lf.select(
            "date",
            "symbol",
            "ts",
            "acc_volume",
            "acc_amount",
            "weighted_mid_mean",
            "spread_bps_mean",
        )
        .collect(engine="streaming")
        .sort("symbol", "ts")
    )
    return (
        df.with_columns(
            volume=pl.col("acc_volume").diff().over("symbol", "date").clip(lower_bound=0),
            amt=pl.col("acc_amount").diff().over("symbol", "date").clip(lower_bound=0),
        )
        .with_columns(
            price=pl.when(pl.col("volume") > 0)
            .then(pl.col("amt") / pl.col("volume"))
            .otherwise(pl.col("weighted_mid_mean")),
            spread_bps=pl.col("spread_bps_mean").cast(pl.Float64),
        )
        .rename({"date": "session"})
        .with_columns(pl.col("session").cast(pl.Utf8))
        .select("session", "symbol", "ts", "price", "volume", "spread_bps")
    )


def main() -> None:
    symbols = [
        "00700",
        "00005",
        "00388",
        "01211",
        "01810",
        "00883",
        "00941",
        "02318",
        "01299",
        "09988",
    ]
    bars = load_bars(symbols, "20260601", "20260831")
    preds = (
        pl.read_parquet(PREDS / "exec_preds_mse_s0_n100_seq60_h128.parquet")
        .rename({"date": "session"})
        .with_columns(pl.col("session").cast(pl.Utf8))
        .select("session", "symbol", "ts", "p")
    )
    signal = load_signal(preds.filter(pl.col("symbol").is_in(symbols)))

    specs = {
        "uniform": {"schedule": "uniform"},
        "static_vwap_20d": {"schedule": "static", "curve_lookback": 20},
        "inst_vwap": {
            "schedule": "inst",
            "curve_lookback": 30,
            "weekday_conditioned": True,
            "weekday_col": "session",
        },
        "inst_vwap_tilt02": {
            "schedule": "inst",
            "curve_lookback": 30,
            "weekday_conditioned": True,
            "weekday_col": "session",
            "kappa": 0.02,
        },
    }
    results = simulate_many(bars, specs, tilt_signal=signal)

    rows = {}
    for name, res in results.items():
        s = summarize_all(res)
        rows[name] = s
        for side, r in s["sides"].items():
            print(
                f"{name:22s} {side}: IS_day {r['is_day']['mean_bp']:+8.3f} "
                f"(t={r['is_day']['t']:+.1f})  IS_arr {r['is_arr']['mean_bp']:+8.3f}  "
                f"agg {r['agg_frac']:.2f}"
            )
    gain_buy = (
        rows["inst_vwap"]["sides"]["buy"]["is_day"]["mean_bp"]
        - rows["inst_vwap_tilt02"]["sides"]["buy"]["is_day"]["mean_bp"]
    )
    print(f"\ntilt gain (buy):  {gain_buy:+.3f} bp/day  (positive = tilt better)")
    write_report(
        rows["inst_vwap_tilt02"],
        str(Path(__file__).resolve().parent / "outputs/exec_example_report.md"),
        title="HK 1-min execution: inst VWAP + tilt (kappa=0.02)",
    )


if __name__ == "__main__":
    main()
