# Parallel Pricing Architecture Comparison

## Overview

A comparison of two approaches to pricing 70,000 structured instruments (~945k legs) across 400 underlyings with delta, gamma sensitivities and P&L.

---

## Approach A — In-Memory Vectorised Pipeline (Python)

| Component | Implementation |
|---|---|
| Market Data | Pre-fetched, held in-memory as dict |
| Pricing Engine | NumPy vectorised Black-Scholes per underlying |
| Parallelism | `ProcessPoolExecutor` — 1 task per underlying |
| Intermediate Storage | In-memory buffers, overflow to local JSON |
| Rollup | `np.add.at` across NumPy arrays |
| Persistence | Final write only (JSON layers) |

### Pipeline

```
Market Data (in-memory)
    │
    ▼
┌──────────────────────────────┐
│  ProcessPoolExecutor (N cpus)│
│  400 tasks (1 per underlying)│
│  Vectorised leg pricing      │
└──────────────┬───────────────┘
               │ results held in RAM
               ▼
┌──────────────────────────────┐
│  Instrument Rollup           │
│  np.add.at (leg → instrument)│
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Portfolio Rollup & Summary  │
│  Write final JSON layers     │
└──────────────────────────────┘
```

**Result: ~7 seconds end-to-end on a 16-core machine.**

---

## Approach B — Java Spring Boot + MongoDB Middleware

| Component | Implementation |
|---|---|
| Market Data | REST API calls per underlying (network-bound) |
| Pricing Engine | Java quant library (Monte Carlo / PDE solvers) |
| Parallelism | 10 portfolio services dispatching jobs |
| Intermediate Storage | MongoDB (read/write per leg batch) |
| Rollup | Read back from MongoDB, aggregate |
| Persistence | MongoDB throughout the pipeline |

### Pipeline

```
Market Data API (HTTP per ticker)
    │  ← network round-trips, rate limits
    ▼
┌──────────────────────────────────┐
│  10 Spring Boot portfolio services│
│  Each submits pricing jobs        │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Java Quant Engine               │
│  Per-instrument pricing          │
│  (heavier models, longer compute)│
└──────────────┬───────────────────┘
               │ write results
               ▼
┌──────────────────────────────────┐
│  MongoDB (intermediate layer)    │  ← serialisation, network I/O,
│  945k leg-level documents        │     write contention across 10 ports
└──────────────┬───────────────────┘
               │ read back
               ▼
┌──────────────────────────────────┐
│  Rollup Service                  │
│  Aggregate from MongoDB          │
│  Write summary back to MongoDB   │
└──────────────────────────────────┘
```

**Result: ~20 minutes per job (10 ports running sequentially ≈ 200 minutes total).**

---

## Where the Time Goes

| Stage | Approach A | Approach B | Why the Gap |
|---|---|---|---|
| Market data fetch | ~0s (in-memory) | 1–2 min | HTTP calls per ticker, rate limits |
| Leg pricing | ~6s | 8–12 min | Vectorised NumPy vs per-instrument Java quant engine |
| Intermediate I/O | ~0s (RAM) | 5–8 min | MongoDB write/read of leg-level documents |
| Rollup | ~0.2s | 2–3 min | NumPy array ops vs MongoDB aggregation pipeline |
| **Total** | **~7s** | **~20 min** | **~170x slower** |

---

## Root Cause Analysis

### 1. MongoDB as a Synchronous Pipeline Gate

Using MongoDB as intermediate storage between compute stages turns a **CPU-bound** problem into an **I/O-bound** one. Each stage must:

- Serialise results (Java objects → BSON)
- Transmit over network to MongoDB cluster
- Wait for write acknowledgement
- Next stage reads back, deserialises

With 10 portfolio services writing concurrently, write contention and index updates compound the latency.

### 2. Per-Instrument vs Per-Underlying Dispatch

Dispatching 70,000 pricing jobs (one per instrument) creates far more overhead than 400 jobs (one per underlying). Many instruments share the same underlyings — grouping by underlying enables vectorised batch pricing of all legs at once.

### 3. Network-Bound Market Data

Fetching market data via REST API introduces latency that doesn't exist when data is pre-loaded. Even with connection pooling, 400+ HTTP calls serialise poorly.

### 4. Heavier Pricing Models

Full Monte Carlo or PDE-based pricing per instrument is legitimately slower than closed-form Black-Scholes. This is the one area where the time difference is partially justified.

---

## Recommendations

| Change | Expected Impact | Effort |
|---|---|---|
| **Batch market data** — single API call for all tickers on the pricing date | Minutes → seconds | Low |
| **Group by underlying** — price all legs per underlying in one vectorised batch | Fewer engine invocations, better cache locality | Medium |
| **Replace MongoDB intermediates with in-memory store** (Redis, Hazelcast, or shared heap) | Eliminates 60–70% of I/O wait | Medium |
| **Async persistence** — write to MongoDB fire-and-forget after pricing, not as a pipeline gate | Decouples compute from I/O | Low |
| **Reduce MongoDB writes** — persist only instrument-level rollups (70k docs), not leg-level (945k docs) | 13x fewer writes | Low |
| **Vectorise in Java** — use `DoubleStream`, SIMD intrinsics, or call into native NumPy via GraalPy | Faster per-leg compute | High |

### Realistic Target

With the above changes applied to the Java Spring Boot stack:

| Scenario | Estimated Time |
|---|---|
| Current state | ~20 min |
| Batch market data + async Mongo writes | 10–12 min |
| + Group by underlying + in-memory intermediates | 3–5 min |
| + Vectorised pricing (or simplified models where applicable) | 1–2 min |

---

## Summary

The 6-second Python result benefits from zero network I/O, closed-form pricing, and in-memory intermediates. A production Java Spring Boot system will always be slower due to real market data feeds and richer pricing models — but the **architecture** should not add more than 10–20x overhead. At ~170x (20 min vs 7s), a significant portion of the time is still attributable to the middleware pattern rather than the compute itself.

**MongoDB is an excellent persistence layer. It is not an efficient message bus between compute stages.**
