"""execsim — a market-agnostic execution simulator for VWAP-style algorithms.

Input contract (one bar per row, any bar size -- 1-min, 5-min, daily slices):

    session    : any hashable session id (date or string), chronological
    symbol     : instrument id
    ts         : bar timestamp (used only for ordering within a session)
    price      : the bar's traded VWAP (or your best fill-price proxy)
    volume     : the bar's traded volume
    spread_bps : optional; bar spread in bp. Default 2.0 when absent. Drives
                 the all-in arrival-price accounting only.

    signal (optional, separate frame): session, symbol, ts, p -- p is a
    *return* forecast for the NEXT bar, made at the close of bar ts. It is
    consumed with a one-bar lag (the forecast for bar b is p at bar b-1), so
    no threshold or alignment work is needed from the caller.

One call:

    from execsim import simulate, simulate_many, summarize_all, write_report

    res = simulate(bars, schedule="inst", order_frac=0.01,
                   tilt_signal=signal_df, kappa=0.02)
    summary = summarize_all(res)          # nested dict of stats
    write_report(summary, "EXEC_REPORT.md")

See docs/CONCEPTS.md for concepts, assumptions, algorithms, and metrics math.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

__all__ = [
    "load_signal",
    "prepare_bars",
    "simulate",
    "simulate_many",
    "summarize",
    "summarize_all",
    "volume_curve",
    "write_report",
]

REQUIRED = ("session", "symbol", "ts", "price", "volume")
_SIDES = ("buy", "sell")

# Fill-model scenarios. "ideal" reproduces the bar-VWAP fill assumption.
# All others add the three realism terms documented in docs/CONCEPTS.md 2:
#   lam   : fraction of the half-spread captured by passive fills
#   pf    : probability a passive order fills within its bar (unfilled size
#           crosses at the half-spread)
#   theta : fraction of the predicted 1-min drift realized AFTER the average
#           fill time of its bar (intra-bar timing decay of the signal)
#   Y     : square-root impact coefficient,
#           impact_bp = Y * sigma_1min_bp * sqrt(participation_in_bar)
FILL_SCENARIOS: dict[str, dict] = {
    "ideal": {"lam": 0.4, "pf": 1.0, "theta": 1.0, "Y": 0.0},
    "optimistic": {"lam": 0.4, "pf": 0.8, "theta": 0.7, "Y": 0.5},
    "neutral": {"lam": 0.2, "pf": 0.5, "theta": 0.5, "Y": 0.75},
    "pessimistic": {"lam": 0.0, "pf": 0.3, "theta": 0.3, "Y": 1.0},
}


# ---------------------------------------------------------------- data prep


def prepare_bars(bars: pl.DataFrame) -> pl.DataFrame:
    """Validate and normalize the bar frame; assign per-session bar ordinals."""
    missing = [c for c in REQUIRED if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    out = bars.sort("symbol", "ts")
    if "spread_bps" not in out.columns:
        out = out.with_columns(pl.lit(2.0).cast(pl.Float64).alias("spread_bps"))
    return out.with_columns(bidx=pl.int_range(pl.len()).over("symbol", "session").cast(pl.Int64))


def volume_curve(  # noqa: ANN001 -- session id is any hashable
    prep: pl.DataFrame,
    session,  # noqa: ANN001
    lookback: int,
    weekday_conditioned: bool = False,
    weekday_col: str | None = None,
) -> pl.DataFrame:
    """Per-symbol volume shares (symbol, bidx, h) over strictly earlier sessions.

    Bars are aligned by ordinal; sessions with different bar counts contribute
    their available bars and the curve is renormalized per symbol.
    """
    nb = prep.filter(pl.col("session") == session).group_by("symbol").agg(n=pl.len())["n"].max()
    sessions = prep["session"].unique().sort().to_list()
    past = sessions[: sessions.index(session)][-lookback:]
    if weekday_conditioned and weekday_col is not None:
        wd = prep.filter(pl.col("session") == session)[weekday_col][0]
        past_wd = [s for s in past if prep.filter(pl.col("session") == s)[weekday_col][0] == wd]
        if len(past_wd) >= 3:
            past = past_wd
    if not past:
        return pl.DataFrame(schema={"symbol": pl.String, "bidx": pl.Int64, "h": pl.Float64})
    hist = prep.filter(pl.col("session").is_in(past))
    tot = hist.group_by("symbol", "session").agg(v=pl.col("volume").sum())
    out = (
        hist.join(tot, on=["symbol", "session"])
        .with_columns(
            sh=pl.when(pl.col("v") > 0).then(pl.col("volume") / pl.col("v")).otherwise(None)
        )
        .group_by("symbol", "bidx")
        .agg(h=pl.col("sh").mean())
    )
    out = out.with_columns(h=pl.col("h").fill_nan(0.0).fill_null(0.0))
    ssum = out.group_by("symbol").agg(s=pl.col("h").sum())
    out = (
        out.join(ssum, on="symbol")
        .with_columns(
            h=pl.when(pl.col("s") > 0).then(pl.col("h") / pl.col("s")).otherwise(1.0 / nb)
        )
        .select("symbol", "bidx", "h")
    )
    return out


def load_signal(signal: pl.DataFrame) -> pl.DataFrame:
    """Validate a signal frame (session, symbol, ts, p). No shifting: the
    simulator consumes p with its own one-bar lag."""
    missing = [c for c in ("session", "symbol", "ts", "p") if c not in signal.columns]
    if missing:
        raise ValueError(f"signal missing required columns: {missing}")
    return signal.sort("symbol", "ts")


# ---------------------------------------------------------------- simulation


def _run_symbol_day(
    h: np.ndarray,
    px: np.ndarray,
    spr: np.ndarray,
    sig: np.ndarray,
    vol: np.ndarray,
    side: str,
    order: float,
    part: float,
    kappa: float,
    inst: bool,
    ema: float,
    passive_credit: float,
    fm: dict,
    sigma_bp: float,
) -> dict | None:
    nb = len(h)
    sgn = 1.0 if side == "buy" else -1.0
    cum_curve = np.cumsum(h)
    Q = order
    filled = num = rnum = den = 0.0
    agg_qty = pas_qty = 0.0
    vhat = 0.0
    p_prev = 0.0
    prev_day_vol = float(np.nansum(np.nan_to_num(vol)))
    for b in range(nb):
        if Q <= 1e-14:
            break
        m = float(np.clip(sgn * kappa * fm["theta"] * p_prev * 1e4, -0.69, 0.69)) if kappa else 0.0
        rem = 1.0 - (cum_curve[b - 1] if b > 0 else 0.0)
        slice_qty = Q * h[b] / max(rem, 1e-12) * float(np.exp(m))
        vhat_eff = prev_day_vol
        if inst and b > 0 and cum_curve[b - 1] >= 0.05:
            realized = float(np.nansum(np.nan_to_num(vol[:b])))  # strictly < b
            new_vhat = ema * vhat + (1 - ema) * (realized / cum_curve[b - 1])
            vhat = new_vhat if np.isfinite(new_vhat) else vhat
            vhat_eff = vhat if vhat > 0 else prev_day_vol
        cap = part * vhat_eff * h[b] if inst else float("inf")
        take = min(slice_qty, max(cap, 0.0))
        target_prev = order * (cum_curve[b - 1] if b > 0 else 0.0)
        agg = min(max(target_prev - filled, 0.0), take) if inst else take
        pas = take - agg
        px_b = px[b]
        if np.isfinite(px_b) and take > 0:
            half = 0.5 * (spr[b] if np.isfinite(spr[b]) else 2.0) / 1e4 * px_b
            part_b = take * vhat_eff / max(vol[b], 1.0)
            imp = fm["Y"] * sigma_bp * float(np.sqrt(min(part_b, 1.0))) / 1e4 * px_b
            pas_fill = pas * fm["pf"]  # filled passively this bar
            pas_cross = pas - pas_fill  # forced to cross (still behind)
            lam = fm["lam"]
            num += (
                agg * (px_b + sgn * (half + imp))
                + pas_fill * (px_b - sgn * lam * half + sgn * imp)
                + pas_cross * (px_b + sgn * (half + imp))
            )
            rnum += take * px_b
            den += take
            filled += take
            Q -= take
            agg_qty += agg
            pas_qty += pas
        p_prev = sig[b]
    if den <= 0 or filled / order < 0.99:
        return None
    return {
        "exec": num / den,
        "raw_exec": rnum / den,
        "completion": filled / order,
        "agg_frac": agg_qty / max(agg_qty + pas_qty, 1e-12),
    }


def simulate(
    bars: pl.DataFrame,
    schedule: str = "inst",
    order_frac: float = 0.01,
    part_cap: float = 0.10,
    curve_lookback: int = 20,
    weekday_conditioned: bool = False,
    weekday_col: str | None = None,
    tilt_signal: pl.DataFrame | None = None,
    kappa: float = 0.0,
    passive_credit: float = 0.4,
    ema: float = 0.3,
    fill_model: str = "ideal",
    sides: tuple[str, ...] = _SIDES,
) -> dict:
    """Simulate one strategy over all sessions. Returns fills + config.

    schedule: "uniform" | "static" (trailing volume curve) | "inst"
    (institutional: weekday-conditioned curve, volume-clock re-sync, urgency
    pacing, passive/aggressive spread accounting). The signal tilt requires
    schedule="inst".

    fill_model: "ideal" (bar-VWAP fills, no impact -- the original assumption),
    or "optimistic" / "neutral" / "pessimistic" bracket scenarios adding
    passive-fill risk, intra-bar signal decay and square-root impact (see
    FILL_SCENARIOS).
    """
    if schedule not in ("uniform", "static", "inst"):
        raise ValueError(f"unknown schedule {schedule!r}")
    if kappa and schedule != "inst":
        raise ValueError("signal tilt requires schedule='inst'")
    fm = FILL_SCENARIOS.get(fill_model)
    if fm is None:
        raise ValueError(f"unknown fill_model {fill_model!r}; one of {sorted(FILL_SCENARIOS)}")
    prep = prepare_bars(bars)
    sig_df = load_signal(tilt_signal) if tilt_signal is not None else None
    sessions = prep["session"].unique().sort().to_list()
    fills: list[dict] = []
    for session in sessions:
        sb = prep.filter(pl.col("session") == session)
        curve = (
            None
            if schedule == "uniform"
            else volume_curve(prep, session, curve_lookback, weekday_conditioned, weekday_col)
        )
        for symbol in sb["symbol"].unique().to_list():
            g = sb.filter(pl.col("symbol") == symbol).sort("ts")
            nb = g.height
            if nb < 2:
                continue
            px = g["price"].to_numpy().astype(np.float64)
            vol = g["volume"].to_numpy().astype(np.float64)
            spr = g["spread_bps"].to_numpy().astype(np.float64)
            h = np.full(nb, 1.0 / nb) if schedule == "uniform" else _curve_for(curve, symbol, nb)
            if h is None or not np.isfinite(h).all() or h.sum() <= 0:
                h = np.full(nb, 1.0 / nb)
            if sig_df is not None:
                gs = sig_df.filter(
                    (pl.col("symbol") == symbol) & (pl.col("session") == session)
                ).sort("ts")
                p_arr = (
                    g.join(gs.select("ts", "p"), on="ts", how="left")["p"]
                    .to_numpy()
                    .astype(np.float64)
                )
                p_arr = np.nan_to_num(p_arr, nan=0.0)
            else:
                p_arr = np.zeros(nb)
            dvwap = float(
                np.nansum(px * np.nan_to_num(vol)) / max(np.nansum(np.nan_to_num(vol)), 1e-12)
            )
            arrival = float(px[0])
            for side in sides:
                o = _run_symbol_day(
                    h,
                    px,
                    spr,
                    p_arr,
                    vol,
                    side,
                    order_frac,
                    part_cap,
                    kappa,
                    schedule == "inst",
                    ema,
                    passive_credit,
                )
                if o is None:
                    continue
                if side == "buy":
                    is_day = (o["raw_exec"] / dvwap - 1) * 1e4
                    is_arr = (o["exec"] / arrival - 1) * 1e4
                else:
                    is_day = (1 - o["raw_exec"] / dvwap) * 1e4
                    is_arr = (1 - o["exec"] / arrival) * 1e4
                fills.append(
                    {
                        "session": session,
                        "symbol": symbol,
                        "side": side,
                        "is_day": is_day,
                        "is_arr": is_arr,
                        "completion": o["completion"],
                        "agg_frac": o["agg_frac"],
                    }
                )
    schema = {
        "session": pl.String,
        "symbol": pl.String,
        "side": pl.String,
        "is_day": pl.Float64,
        "is_arr": pl.Float64,
        "completion": pl.Float64,
        "agg_frac": pl.Float64,
    }
    fdf = pl.DataFrame(fills) if fills else pl.DataFrame(schema=schema)
    return {
        "fills": fdf,
        "config": {
            "schedule": schedule,
            "order_frac": order_frac,
            "part_cap": part_cap,
            "curve_lookback": curve_lookback,
            "weekday_conditioned": weekday_conditioned,
            "kappa": kappa,
            "passive_credit": passive_credit,
            "ema": ema,
            "n_sessions": len(sessions),
        },
    }


def _curve_for(curve: pl.DataFrame | None, symbol: str, nb: int) -> np.ndarray | None:
    if curve is None or curve.height == 0:
        return None
    c = curve.filter(pl.col("symbol") == symbol).sort("bidx")
    if c.height == 0:
        return None
    h = np.zeros(nb)
    jj = c["bidx"].to_numpy().astype(int)
    ok = jj < nb
    h[jj[ok]] = c["h"].to_numpy()[ok]
    if h.sum() <= 0:
        return None
    return h / h.sum()


# ---------------------------------------------------------------- reporting


def _side_stats(fills: pl.DataFrame, col: str) -> dict:
    v = fills[col].to_numpy()
    sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    t = float(v.mean() / (sd / np.sqrt(len(v)))) if len(v) > 1 and sd > 0 else None
    return {"mean_bp": float(v.mean()), "std_bp": sd, "t": t, "n": int(len(v))}


def summarize(fills: pl.DataFrame, side: str) -> dict:
    f = fills.filter(pl.col("side") == side)
    if f.height == 0:
        return {}
    return {
        "is_day": _side_stats(f, "is_day"),
        "is_arr": _side_stats(f, "is_arr"),
        "completion": float(f["completion"].mean()),
        "agg_frac": float(f["agg_frac"].mean()),
    }


def summarize_all(res: dict) -> dict:
    out = {"config": res["config"], "sides": {}}
    for side in res["fills"]["side"].unique().to_list():
        out["sides"][side] = summarize(res["fills"], side)
    return out


def simulate_many(
    bars: pl.DataFrame,
    specs: dict[str, dict],
    tilt_signal: pl.DataFrame | None = None,
) -> dict[str, dict]:
    """Run several configs on identical bars. Each spec is a kwargs dict for
    simulate(); a spec may override "tilt_signal"."""
    out = {}
    for name, kw in specs.items():
        sig = kw.pop("tilt_signal", tilt_signal)
        out[name] = simulate(bars, tilt_signal=sig, **kw)
    return out


def _fmt(r: dict, key: str) -> str:
    v = r[key]
    t = v["t"]
    ts = f"(t={t:+.1f})" if t is not None else ""
    return f"{v['mean_bp']:+.3f} {ts}"


def write_report(summary: dict, path: str | Path, title: str = "Execution report") -> None:
    lines = [f"# {title}", ""]
    c = summary.get("config", {})
    lines += [
        f"- schedule: `{c.get('schedule')}`, kappa={c.get('kappa')}, "
        f"order_frac={c.get('order_frac')}, part_cap={c.get('part_cap')}, "
        f"lookback={c.get('curve_lookback')}, sessions={c.get('n_sessions')}",
        "",
        "| side | IS_day bp | IS_arr bp | completion | agg frac | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for side, r in summary.get("sides", {}).items():
        lines.append(
            f"| {side} | {_fmt(r, 'is_day')} | {_fmt(r, 'is_arr')} "
            f"| {r['completion']:.3f} | {r['agg_frac']:.2f} | {r['is_day']['n']} |"
        )
    lines += [
        "",
        "IS positive = cost. IS_day vs the day's VWAP (ex-post, "
        "drift-contaminated -- compare strategies, not levels). IS_arr is "
        "all-in vs arrival price incl. spread accounting.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
