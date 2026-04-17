# Production Data Flow: Structured Products Pricing at Scale

## Overview

End-to-end data flow analysis for pricing 2.1M structured instruments (30 portfolios × 70,000 instruments) across 400 underlyings with full sensitivities (delta, gamma, vega, theta, rho, cross-gamma) and P&L — covering Monte Carlo simulation, closed-form pricing, and hierarchical rollup.

This document examines the optimal architecture used by Tier 1 banks and arbitrage desks, and contrasts it with a traditional Java Spring Boot + MongoDB middleware approach.

---

## The 7-Stage Pipeline

### Stage 1: Trade Collection

The most underestimated upstream dependency. Before anything prices, the system must assemble **what to price**.

**Responsibilities:**
- Pull live portfolio positions from the booking system (e.g. Murex, Calypso, Summit, or internal OMS)
- Resolve instrument definitions — leg structures, underlying mappings, barrier levels, autocall schedules
- Snapshot trade state as of pricing cutoff (T, T-1, or intraday)
- Deduplicate across desks (same underlying, overlapping instruments across books)
- Produce the canonical pricing request: instruments + legs + underlyings + trade metadata

At Tier 1 bank scale (millions of instruments across rates, equities, FX, credit), trade collection alone can take 5–15 minutes if it queries relational booking systems serially.

**Optimal pattern:** Pre-assemble into a Kafka topic or in-memory grid so pricing consumers have data ready at cutoff.

---

### Stage 2: Market Data Assembly

**Responsibilities:**
- Parallel bulk fetch: spots, vol surfaces (thousands of strike/expiry pairs), rate curves (30+ tenors), dividends, correlations, repo rates
- One call per data type, not per ticker
- Construct an immutable market data snapshot for the pricing date

**Storage:** Shared memory or broadcast to all workers.

At bank scale, market data lives in an in-memory grid (e.g. Gemfire, Coherence, or proprietary) that's already populated by upstream feeds. The pricing service doesn't "fetch" — it reads from shared memory. Cost: **near-zero latency**.

**Bottleneck:** Vol surface construction (interpolation across strike/expiry grid).

---

### Stage 3: Scenario Generation

**Responsibilities:**
- Build bump scenarios: base, spot ±1%, vol ±1%, rate ±1bp, time ±1d, cross-gamma, etc.
- For Monte Carlo: generate correlated paths via Cholesky decomposition on the correlation matrix across 400 underlyings
- Output a scenario tensor: `N_scenarios × N_underlyings × N_timesteps`

**Storage:** GPU VRAM or shared memory.

**Bottleneck:** Correlation matrix decomposition for 400 underlyings (400×400 matrix).

---

### Stage 4: Pricing (The Compute Core)

This is where the architectural decision matters most.

```
┌─────────────────────────────────────────────────┐
│           GROUP BY UNDERLYING                   │
│                                                 │
│  NOT by instrument. This is the key             │
│  architectural decision.                        │
│                                                 │
│  Each underlying task prices ALL legs           │
│  across ALL instruments referencing it,         │
│  under ALL scenarios, in one vectorised pass.   │
└─────────────────────────────────────────────────┘
```

**Pricing methods by instrument type:**

| Instrument Type | Pricing Method | Paths Needed | Per-Instrument Cost |
|---|---|---|---|
| Vanilla European | Black-Scholes (closed-form) | 0 | ~microseconds |
| Barrier / Knockout | Monte Carlo | 100k–500k | 10–100ms CPU, 0.1–1ms GPU |
| Autocallable | Monte Carlo (path-dependent) | 500k–2M | 50–500ms CPU, 1–5ms GPU |
| Worst-of basket (4-5 underlyings) | MC with correlated paths | 500k–2M × 5 UL | 200ms–2s CPU, 2–20ms GPU |
| American/Bermudan | Longstaff-Schwartz MC or PDE | 200k + regression | 100ms–1s CPU, 5–50ms GPU |

For 70,000 instruments with 4-5 underlyings each, many being autocallables or worst-of baskets, **Monte Carlo is the dominant cost**. A single autocallable with 5 underlyings, 1M paths, and 60 monthly observation dates requires pricing 60M path-steps — per instrument. Across 70k instruments that's trillions of floating-point operations.

**Output:** Leg-level PV, delta, gamma, vega, theta, rho per scenario.

**Storage:** IN MEMORY. Not MongoDB. Not disk. Compute stages must never leave the process/GPU.

---

### Stage 5: Instrument Rollup

**Responsibilities:**
- Scatter-add leg results → instrument-level Greeks & P&L
- Cross-gamma: Δ(delta_A) when bumping underlying_B
- Correlation risk: how instruments with shared underlyings interact

**Output:** Instrument-level risk vector.

**Storage:** In-memory arrays.

**Bottleneck:** Minimal — pure aggregation (O(n) scatter-add).

---

### Stage 6: Book / Desk / Portfolio Rollup

**Responsibilities:**
- Instrument → book → desk → legal entity → firm-wide hierarchy
- Netting: long/short offsets within same underlying
- Concentration risk: exposure per underlying, per sector
- VaR / CVaR: portfolio-level using scenario P&L distribution

**Output:** Hierarchical risk report.

**Storage:** In-memory → then persist.

**Bottleneck:** VaR simulation if full historical revaluation.

---

### Stage 7: Persistence & Reporting

**This is the ONLY stage that touches disk or network storage.**

- Write final results to database (MongoDB, PostgreSQL, kdb+)
- Push to risk dashboards, trader blotters, regulatory feeds
- Archive scenario detail for audit trail

---

## Correlation Grouping: The Shared-Path Optimisation

The "group by underlying" pattern extends to **correlation groups** in production:

- Instruments with the same basket of underlyings share the same correlated path simulation
- 70,000 instruments might only have 5,000–10,000 unique underlying baskets
- Generate correlated paths once per basket, price all instruments referencing that basket

Per-instrument dispatch to a database (the traditional middleware approach) misses this optimisation entirely — it treats each instrument as independent, duplicating path generation for every instrument sharing the same basket.

---

## Sensitivity Computation Methods

| Method | Approach | Cost (for 6 Greeks) | Used By |
|---|---|---|---|
| **Full revaluation** | Price at base, price at each bump, take difference | 13× base pricing (1 base + 6 bumps × 2 for central differences) | Simpler setups |
| **Adjoint Algorithmic Differentiation (AAD)** | Compute all Greeks in one forward+backward pass | ~2–4× base pricing for ALL Greeks simultaneously | JPM, Goldman, most Tier 1 banks |
| **Pathwise sensitivities** | Differentiate through MC paths analytically | ~1× base pricing for delta/gamma/vega/rho together | Quant hedge funds |

AAD gives all 6 Greeks for ~3× base pricing compared to 13× for full revaluation. At 70k instruments that's a **4× speedup** in the most expensive stage.

---

## Intermediate Storage: What Goes Where

| Pattern | Use Case | Latency | Verdict |
|---|---|---|---|
| MongoDB between stages | Audit, replay, fault tolerance | 1–10ms per doc | **Only for final persistence (Stage 7)** |
| Redis / Valkey | Cross-service sharing, caching | 0.1–1ms per key | Good for market data cache |
| Shared memory (mmap) | Same-machine pipeline stages | ~microseconds | Best for pricing → rollup |
| Kafka | Event-driven, multi-consumer | 1–5ms per message | Good for trade collection → pricing dispatch |
| In-process arrays | Single pipeline | 0 | Best for compute-bound stages (4, 5, 6) |

**The rule:** Compute stages (4, 5, 6) should NEVER leave the process/GPU. Persistence happens once at Stage 7.

---

## Arbitrage Desk vs Bank Desk

| Dimension | Arbitrage / Prop Desk | Bank Desk (Tier 1) |
|---|---|---|
| Instrument count | 5k–50k (focused strategies) | 500k–5M (full client book) |
| Latency requirement | Sub-second (exploit mispricings) | 5–20 min acceptable (EOD risk) |
| Re-pricing frequency | Continuous / every tick | EOD + 2–4 intraday snapshots |
| Market data freshness | Real-time streaming | Snapshot at cutoff time |
| Monte Carlo precision | Lower (faster decisions) | Higher (regulatory requirement) |
| Infrastructure | Single GPU server, lean | Hundreds of compute nodes, GPU farm |
| Intermediate storage | None — all in-memory | Required for audit/regulatory |

An arbitrage desk with this setup (70k instruments, 400 underlyings) would run the entire pipeline on **one g5.xlarge** instance and reprice every few seconds. A bank desk with 2M instruments would distribute across a **GPU cluster** (16–64 A10G/A100 nodes) and target 10-minute cycles.

---

## Optimal Architecture for 30 Jobs × 70k Instruments

```
Trade Collection (Kafka topic, pre-assembled)
       │
       ▼
Market Data Grid (in-memory, bulk snapshot)
       │
       ▼
Scenario Generator (GPU: Cholesky + path gen)
       │  broadcast scenario tensor to pricing
       ▼
┌──────────────────────────────────────────┐
│  GPU Pricing Farm                        │
│  Group by underlying basket              │
│  All 30 portfolios in one mega-batch     │
│  28.4M legs × N_scenarios                │
│  Result: in-GPU-memory                   │
└──────────┬───────────────────────────────┘
           │  no serialisation, no network
           ▼
Rollup (GPU kernel: scatter-add)
           │
           ▼
Persist final rollups → MongoDB / kdb+
Push to dashboards
```

---

## Expected Timing (Optimal Architecture)

| Stage | Time |
|---|---|
| Trade collection (pre-cached) | 0s (Kafka consumer, already ready) |
| Market data snapshot | <1s (in-memory grid read) |
| Scenario generation (GPU) | 0.5–2s (Cholesky + 1M paths × 400 UL) |
| Pricing — closed-form legs | 0.2–0.5s |
| Pricing — Monte Carlo legs | 2–10s (depends on path-dependent instrument %) |
| Rollup (GPU) | 0.05s |
| Persist to DB | 1–3s (async, non-blocking) |
| **Total** | **4–16s for 2.1M instruments** |

---

## Comparison: Optimal vs Middleware Architecture

| Metric | Optimal (GPU + in-memory) | Middleware (Spring Boot + MongoDB) |
|---|---|---|
| 30 portfolios, 2.1M instruments | 4–16s | ~10 hours (30 × 20 min) |
| Intermediate I/O | 0 (in-memory / VRAM) | 28.4M MongoDB document writes + reads |
| Market data fetch | <1s (bulk from grid) | Minutes (per-ticker HTTP) |
| Sensitivity method | AAD or pathwise (3× base) | Full reval (13× base) |
| Infrastructure cost | 1× g5.xlarge ($1.01/hr) | 10+ Spring Boot instances + MongoDB cluster |

The 10-hour figure assumes sequential execution of 30 jobs at 20 min each. Even with parallelism across the 10 portfolio services, the MongoDB contention means the 30 jobs complete in ~60–90 minutes, not the 4–16 seconds achievable with the optimal architecture.

The bottleneck is never the quant math. It's the data flow.
