# Concepts, Assumptions, Algorithms, and Metrics

Reference document for `execsim`. Everything here is implemented in
`execsim.py`; the empirical study that motivated the design is in
the accompanying empirical study (not included in this repo).

---

## 1. Concepts

### 1.1 The problem

A **parent order** of size `Q` (as a fraction of the session's traded volume,
default 1%) must be fully executed across one trading session, split into
slices over `nb` consecutive bars. Buy and sell are simulated separately and
never netted — each side is an independent execution problem.

### 1.2 Volume curve

The **volume curve** `h[b]` is the per-symbol profile of where the session's
volume historically trades: `h[b] = vol_b / Σ_b vol_b`, averaged over the
`lookback` (default 20) **strictly earlier** sessions, optionally conditioned
on weekday (mean over same-weekday sessions within the pool; fallback to all
sessions when fewer than 3 exist). Bars are aligned by ordinal position within
the session. The curve is renormalized per symbol; symbols with no history
fall back to the uniform curve `h[b] = 1/nb`.

Known limitation (measured in the HK study): a trailing-20 curve correlates
only ~0.4 with the day's *actual* shape — day-to-day shape variation is
mostly idiosyncratic. The curve is a weak prior; the pacing logic, not the
curve, is where execution quality lives.

### 1.3 Benchmarks

| benchmark | definition | role |
|---|---|---|
| **Day VWAP** | `Σ_b price_b · vol_b / Σ_b vol_b` (over the session) | ex-post; a full-information reference. Levels are contaminated by session drift and tracking error — **compare strategies, never quote levels**. |
| **Arrival price** | the first bar's `price` of the session | the desk metric: what the order achieved vs deciding to trade at the open. |

### 1.4 Implementation shortfall (IS)

Per symbol-session, in basis points, positive = cost:

```
buy :  IS = (exec / benchmark − 1) × 10⁴
sell:  IS = (1 − exec / benchmark) × 10⁴
```

Two flavors are reported:

- **IS_day** — benchmark = day VWAP, execution price = *raw* fills (no spread
  accounting). Ex-post and drift-contaminated: in a falling session every buy
  schedule shows negative IS. The deliverable is the **paired difference**
  between strategies on identical orders and sessions.
- **IS_arr** — benchmark = arrival price, execution price = **all-in** (incl.
  spread accounting, §2.3). Contains the session's directional move for any
  full-session order (a buy rides the whole drift), so buy/sell levels are
  asymmetric by construction.

### 1.5 Signal and tilt

A **signal** frame provides `p` at each bar: a forecast of the *next bar's*
return, in dimensionless return units (`3.5e-4` = 3.5 bp), made at the close
of that bar. The simulator consumes it with a **one-bar lag internally**: the
schedule for bar `b` uses `p[b−1]`, which is exactly a forecast for bar b
made at bar b's open. The caller never shifts anything.

The **tilt** multiplies bar b's slice by

```
m_b = exp( κ · sgn · p[b−1] · 10⁴ ),   clipped to ±0.69  (slice ∈ [0.5×, 2×])
```

- `sgn = +1` for buys, `−1` for sells: a predicted up-move pulls buys earlier
  and pushes sells later.
- `κ` is the **volume elasticity per bp of predicted move** (0.01 → ±1%/bp).
- `Σ slice = Q` always: the remaining schedule renormalizes, so κ redistributes
  *when* to trade, never *how much*.
- No thresholding: small predictions produce proportionally small adjustments;
  the clip provides the bound that a threshold would otherwise provide.

Measured (HK study): +0.5 bp/day at κ=0.01, +1.6–2.0 bp/day at κ=0.02, stable
across liquidity tiers, monotone in κ, best with an MSE-trained signal (the
tilt consumes a smooth central drift estimate, not tail conviction).

### 1.6 Causality contract

The schedule for bar `b` may use:
1. volume curves of sessions **strictly earlier** than b's session;
2. realized volume **through bar b−1** (the volume-clock estimate `V̂` and the
   pace check);
3. the signal value `p[b−1]` (a forecast made at bar b−1's close for bar b's
   move).

Bar b's own realized volume/price/spread are used only for its fill, never for
its sizing or pacing decision. Verified empirically in the HK study: the
cross-correlation profile of the signal peaks at the trained horizon and
monotonically accumulates forward — the fingerprint of a causal forecast, not
a shifted one.

---

## 2. Assumptions (all labeled, none hidden)

| # | assumption | where it bites | default |
|---|---|---|---|
| A1 | **fills at bar VWAP** — a slice trades at its bar's traded VWAP | optimistic: no queue, no latency, no impact | always on |
| A2 | **market fills pay ½ spread** (all-in accounting) | price-taker crossing | `spread_bps` per bar |
| A3 | **passive fills earn 0.4 × ½ spread** | the institutional executor's on-pace fills | `passive_credit=0.4` |
| A4 | **participation cap** vs *predicted* bar volume `part · V̂ · h[b]`; shortfall vs the volume clock is crossed | binds only at ≥ ~5% participation; at 1% ADV it never binds (measured: 0 binding bars) | `part=0.10` |
| A5 | **completion forced** — the remaining schedule renormalizes; symbol-days filling <99% are dropped from the sample | report completion; 1.000 in all published runs | threshold 0.99 |
| A6 | **zero latency** between bar close and order action | a real desk eats part of the 1-min drift; live realization < simulated tilt gain | inherent |
| A7 | **no impact model** | fine for ≤1–2% ADV; mandatory before claiming tilt value at larger sizes | — |

---

## 3. Algorithms

### 3.1 Uniform

```
slice_b = Q / nb
```

No information beyond the calendar. Its IS_day level in a trending sample is
pure drift exposure; it exists as the no-information control.

### 3.2 Static volume-curve VWAP

```
slice_b = Q · h[b] / (1 − cum_h[b−1]),     cum_h = Σ_{c<b} h[c]
```

The division by the remaining curve share guarantees Σ slice = Q (a naive
`Q·h[b]` fills only `1−1/e ≈ 63%` — an actual bug caught in the HK study).
Market fills; IS_day ≈ 0 + drift if the curve matches the day's shape.

### 3.3 Institutional VWAP (schedule = "inst")

Practice-style additions on top of a weekday-conditioned curve:

1. **Volume-clock re-sync.** Projected day total, updated once the curve is
   ≥5% complete, from volume realized **through bar b−1**:

   ```
   V̂ ← 0.3·V̂ + 0.7 · ( realized_cum(vol[:b]) / cum_h[b−1] )
   ```

2. **Participation cap** on predicted volume:
   `slice_b ≤ part · V̂ · h[b]`. Causal (uses the forecast, not bar b's
   realized volume). This is what makes the algorithm safe at 5–30%
   participation; at 1% it never binds.

3. **Urgency pacing.** Passive by default; the shortfall vs the **previous**
   bar's target is crossed:

   ```
   target_prev = Q · cum_h[b−1]
   agg_b  = max(target_prev − filled_before_b, 0)      # capped at slice_b
   pas_b  = slice_b − agg_b
   ```

   Effective prices: aggressive at `px + sgn·½·spread`, passive at
   `px − sgn·passive_credit·½·spread`.

4. **Signal tilt** (κ > 0): the exponential multiplier of §1.5 is applied
   inside `slice_b`, before the cap. Tilting ahead of predicted drift keeps
   the executor on-pace, which also *reduces* aggression (measured: 1.00 →
   0.23 aggressive fraction at κ=0.02).

### 3.4 Why the tilt needs a smooth signal

The tilt is a monotone transform of `p`; its value is `cov(p, realized
future drift)`. Tail-focused losses (DBE, CZAR) improve precision on *rare
large moves* but degrade the small-move central estimate that a 1-min
scheduler consumes bar after bar. Measured on five loss-trained models
(HK, top-100): tilt gain mse > czar2 > huber > czar_w > dbe — the inverse of
the tail-metric ranking. Rule of thumb: **MSE-trained signal for scheduling,
asymmetric-loss signal for quoting.**

---

## 4. Evaluation metrics (math)

Per symbol-session, side = buy/sell simulated separately.

```
IS_day[s,sym]  = sgn' · (raw_exec / day_vwap − 1) · 10⁴
IS_arr[s,sym]  = sgn' · (all_in_exec / arrival − 1) · 10⁴
   sgn' = +1 for buys (exec above benchmark = cost),
          −1-fold for sells: IS_sell = (1 − exec/benchmark) · 10⁴
completion    = filled / Q                 (symbol-days < 0.99 are dropped)
agg_frac      = Σ agg / (Σ agg + Σ pas)
```

Aggregation across symbol-days is **day-clustered**: first the mean over
symbols within a session, then across sessions:

```
mean = (1/D) Σ_d mean_{sym} IS
t    = mean / ( std_d(mean_IS) / √D )        # D = number of sessions
IR   = t·√D  (information ratio of the per-day savings)
```

Day-clustering is mandatory: symbol-days within a session share the session's
drift, so treating 4,826 symbol-days as independent inflates t ~6×.

**The deliverable is the paired difference** between two strategies on
identical orders/days:

```
Δ = mean_IS(strategy A) − mean_IS(strategy B)
```

Because both face the same session drift and the same benchmark noise, Δ
isolates scheduling skill (plus any alpha the signal embeds — see the
drift-day decomposition in the HK report: gains concentrated on
trend-aligned days are an alpha bet, not execution).

---

## 5. Worked example

`examples/hk_1min.py` (real HK 1-min data, 10 names, 64 sessions):

```
inst_vwap       buy IS_day −2.175 bp   sell +2.175 bp   completion 1.000
inst_vwap_tilt02 buy −2.650 bp   sell +1.714 bp     tilt gain ≈ +0.47 bp/side
```

Full study (162 sessions, 30–100 names, five losses, three tiers):
the accompanying empirical study (not included in this repo).
