"""Synthetic-data tests: causality, completion, tilt behavior."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from execsim import simulate, summarize_all  # noqa: E402

NB = 60
N_SESSIONS = 8
N_SYM = 3


def make_bars(drift: float = 0.0, seed: int = 0) -> pl.DataFrame:
    """Constant-price sessions with an optional linear drift applied late."""
    rng = np.random.default_rng(seed)
    rows = []
    for si in range(N_SESSIONS):
        session = f"s{si:02d}"
        for sym in range(N_SYM):
            px = 100.0
            for b in range(NB):
                move = drift * (b - NB / 2) / NB  # linear drift across session
                px = 100.0 * (1 + move)
                v = float(rng.uniform(50, 150))
                rows.append(
                    {"session": session, "symbol": f"{sym:03d}", "ts": b, "price": px, "volume": v}
                )
    return pl.DataFrame(rows)


def perfect_signal(bars: pl.DataFrame) -> pl.DataFrame:
    """p at bar b = realized return into bar b+1 (an oracle used ONLY to test
    that the tilt consumes the lagged forecast correctly; not a causality
    claim about real signals)."""
    sub = bars.sort("symbol", "session", "ts").with_columns(
        rf=pl.col("price").shift(-1).over("symbol", "session") / pl.col("price") - 1
    )
    return sub.filter(pl.col("rf").is_finite()).select("session", "symbol", "ts", p=pl.col("rf"))


def test_completion_and_symmetry() -> None:
    res = simulate(make_bars(), schedule="static")
    s = summarize_all(res)
    for side in ("buy", "sell"):
        assert s["sides"][side]["completion"] == pytest.approx(1.0)
    # driftless constant prices -> IS ~ 0 for both sides
    assert abs(s["sides"]["buy"]["is_day"]["mean_bp"]) < 1.0
    assert abs(s["sides"]["sell"]["is_day"]["mean_bp"]) < 1.0


def test_buy_sell_symmetry_drift() -> None:
    bars = make_bars(drift=-0.01)  # prices fall across the session
    s = summarize_all(simulate(bars, schedule="uniform"))
    b = s["sides"]["buy"]["is_day"]["mean_bp"]
    sl = s["sides"]["sell"]["is_day"]["mean_bp"]
    assert b == pytest.approx(-sl, abs=0.5)


def test_tilt_beats_untitled_with_oracle_signal() -> None:
    bars = make_bars(drift=-0.02)
    sig = perfect_signal(bars)
    base = summarize_all(simulate(bars, schedule="inst"))
    tilt = summarize_all(simulate(bars, schedule="inst", tilt_signal=sig, kappa=0.02))
    for side in ("buy", "sell"):
        gain = base["sides"][side]["is_day"]["mean_bp"] - tilt["sides"][side]["is_day"]["mean_bp"]
        assert gain > 0, f"{side}: tilt should improve fills on trend days"


def test_zero_kappa_equals_untitled() -> None:
    bars = make_bars(drift=-0.02)
    sig = perfect_signal(bars)
    a = summarize_all(simulate(bars, schedule="inst", tilt_signal=sig, kappa=0.0))
    b = summarize_all(simulate(bars, schedule="inst"))
    assert a["sides"]["buy"]["is_day"]["mean_bp"] == pytest.approx(
        b["sides"]["buy"]["is_day"]["mean_bp"], abs=1e-9
    )


def test_curve_uses_only_earlier_sessions() -> None:
    """Session 0 must fall back to uniform (no history): its schedule must
    equal uniform's fills."""
    bars = make_bars()
    u = simulate(bars, schedule="uniform")["fills"]
    st = simulate(bars, schedule="static")["fills"]
    u0 = u.filter(pl.col("session") == "s00").sort("symbol", "side")
    s0 = st.filter(pl.col("session") == "s00").sort("symbol", "side")
    assert u0["is_day"].to_list() == pytest.approx(s0["is_day"].to_list(), abs=1e-9)


def test_bad_schedule_raises() -> None:
    with pytest.raises(ValueError):
        simulate(make_bars(), schedule="twap")
