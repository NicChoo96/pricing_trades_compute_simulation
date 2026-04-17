# Parallel Pricing Simulation

A high-performance Python simulation that prices **70,000 structured trade instruments** across **400 active underlyings** (500 mocked), computing delta, gamma sensitivities and P&L — all in under 10 seconds on commodity hardware.

The project demonstrates how vectorised in-memory computation with parallel dispatch compares against a traditional Java Spring Boot + MongoDB middleware architecture for the same workload, and explores RAM-aware offloading strategies for large result sets.

---

## What This Does

- Generates 70,000 instruments, each linked to 4–5 underlyings (with overlap across instruments), producing **~945,000 legs** of type PUT, FUNDING, and COUPON
- Mocks spot, volatility, risk-free rate, and dividend yield for 500 underlying tickers on a single pricing date
- Prices all legs in parallel using a `ProcessPoolExecutor` with one task per underlying — each task vectorises Black-Scholes (PUT legs) and analytical pricing (FUNDING/COUPON) via NumPy
- Computes **delta** and **gamma** sensitivities per leg, plus **P&L** via a 1% spot bump full revaluation
- Monitors RAM usage in real time; when memory pressure exceeds a threshold, intermediate results are flushed to JSON files on disk and transparently read back for rollup
- Rolls up results across three layers: **leg → instrument → underlying → portfolio**
- Displays a live terminal dashboard (Rich) showing pipeline progress, worker activity, throughput, memory pressure, and running portfolio Greeks/P&L
- Outputs ranked tables of top instruments and underlyings by exposure, plus a Taylor expansion P&L validation

---

## Files

### [`simulate_pricing.py`](simulate_pricing.py)

The main script. Runs end-to-end in four phases:

| Phase | What It Does |
|---|---|
| **Layer 0 — Data Generation** | Creates 500 mock underlying market data entries (spot, vol, rate, div yield). Generates 70,000 instruments, each referencing 4–5 of the 400 active underlyings. Each underlying–instrument pair produces 3 legs (PUT, FUNDING, COUPON). Legs are grouped by underlying for efficient dispatch. Market data is persisted to `layer0_market_data.json`. |
| **Layer 1 — Parallel Pricing** | Dispatches 400 tasks (one per underlying) to a process pool. Each task receives all legs for that underlying and prices them in a single vectorised NumPy pass. PUT legs use Black-Scholes closed-form with a spot bump for P&L. FUNDING legs discount notional at the risk-free rate. COUPON legs accrue a fixed coupon discounted to present value. Results are buffered in memory; when RAM exceeds 75% or every 100 underlyings, the buffer is flushed to numbered JSON files (`layer1_batch_NNNN.json`). |
| **Layer 2 — Instrument Rollup** | Reads all Layer 1 results (from memory + disk), then aggregates leg-level delta, gamma, price, and P&L into instrument-level totals using `np.add.at` for O(n) scatter-add across 70,000 instruments. Output saved to `layer2_instrument_rollup.json`. |
| **Layer 3 — Portfolio Rollup** | Aggregates instrument-level results into per-underlying and portfolio-wide totals. Writes the final summary to `layer3_portfolio_summary.json`, including timing metrics. |

**Key design choices:**

- **Group by underlying, not by instrument** — since many instruments share underlyings, grouping legs by underlying minimises redundant market data lookups and enables vectorised pricing of thousands of legs in a single NumPy call
- **Mixed long/short positions** — each underlying–instrument link is randomly assigned long or short (50/50) so portfolio Greeks partially net out, producing realistic book-level numbers
- **Notionals scaled to $1k–$50k** — avoids unrealistically large aggregate P&L while still exercising the full pipeline
- **RAM-aware offloading** — `psutil` monitors system memory; results are kept in-memory when possible for speed, but automatically spill to JSON intermediates when pressure rises, allowing the pipeline to handle workloads larger than available RAM

**Live dashboard** — a Rich `Live` panel refreshes at 12fps showing:
- Pipeline stage ribbon (Generate → Pricing → Rollup → Summary)
- Progress bar with underlying/leg counts
- Throughput (legs/sec) and ETA
- RAM gauge with peak tracking and offload event counter
- Active worker count and recently completed tickers
- Running portfolio-wide Σ Delta, Σ Gamma, Σ P&L

### [`ARCHITECTURE_COMPARISON.md`](ARCHITECTURE_COMPARISON.md)

A detailed comparison between the in-memory Python approach and a traditional Java Spring Boot + MongoDB middleware architecture for the same 70,000-instrument workload. Covers:

- **Pipeline diagrams** for both approaches
- **Time breakdown by stage** — market data fetch, leg pricing, intermediate I/O, rollup
- **Root cause analysis** of why using MongoDB as a synchronous intermediate layer between compute stages turns a CPU-bound problem into an I/O-bound one (~170x slower)
- **Recommendations** with estimated impact — batch market data, group by underlying, replace MongoDB intermediates with in-memory stores, async persistence, vectorised pricing
- **Realistic improvement targets** — from ~20 min down to 1–2 min with architectural changes

### [`requirements.txt`](requirements.txt)

Python dependencies:
- `numpy` — vectorised leg pricing and rollup
- `scipy` — `norm.cdf`/`norm.pdf` for Black-Scholes
- `psutil` — real-time RAM monitoring for offload decisions
- `rich` — live terminal dashboard

### `pricing_intermediates/`

Output directory created at runtime. Contains the layered JSON files:

| File | Contents |
|---|---|
| `layer0_market_data.json` | All 500 underlying market data entries for the pricing date |
| `layer1_batch_NNNN.json` | Leg-level pricing results per underlying (offloaded batches) |
| `layer2_instrument_rollup.json` | Instrument-level aggregated delta, gamma, price, P&L |
| `layer3_portfolio_summary.json` | Portfolio-wide totals and timing metrics |

---

## Quick Start

```bash
pip install -r requirements.txt
python simulate_pricing.py
```

---

## Performance

On a 16-core machine with 64GB RAM:

| Metric | Value |
|---|---|
| Instruments | 70,000 |
| Total legs | ~945,000 |
| Active underlyings | 400 |
| Pricing time | ~6–7s |
| Rollup time | ~0.2s |
| Throughput | ~117,000 legs/sec |
| Peak RAM | ~39% of 64GB |
| JSON offloads | 4 batches (~67 MB) |
