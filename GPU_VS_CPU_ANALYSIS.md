# GPU vs CPU Pricing Analysis

## Workload Definition

30 parallel jobs, each pricing 70,000 structured instruments across 400 active underlyings.

| Metric | Per Job | Total (30 Jobs) |
|---|---|---|
| Instruments | 70,000 | 2,100,000 |
| Legs (PUT + FUNDING + COUPON) | ~945,000 | ~28,400,000 |
| Active underlyings | 400 | 400 (shared market data) |
| Market data entries | 500 | 500 (same pricing date) |

---

## Execution Time

| Strategy | Config | Time | Notes |
|---|---|---|---|
| **CPU sequential** | 16-core instance, 1 job at a time | 30 × 7s = **~3.5 min** | Simple, low RAM |
| **CPU parallel (all 30)** | 96+ cores, all jobs concurrent | **~10–15s** | Needs ~42GB extra RAM (30 × 1.4GB) |
| **CPU parallel (6 batches of 5)** | 32-core instance | 6 × 7s = **~42s** | Balanced RAM/speed |
| **GPU sequential** | 1× A10G, one job at a time | 30 × 250ms = **~7.5s** | 24GB VRAM, plenty |
| **GPU mega-batch** | 1× A10G, all 28.4M legs in one kernel | **~1–3s** | ~4GB VRAM needed |
| **GPU mega-batch** | 1× A100 (80GB) | **~0.5–1.5s** | Overkill but fastest |

---

## Why This Workload Is Ideal for GPU

- **945k legs per job are independent** at the pricing stage — textbook SIMD parallelism
- **Black-Scholes is pure arithmetic** (`exp`, `log`, `sqrt`, `erf`) — GPUs have dedicated hardware for transcendental functions
- **Broadcast-friendly memory access** — all legs for one underlying share the same spot/vol/rate, so a single memory read fans out to thousands of cores
- **Dense float64 arrays, no branching** — exactly what GPU cores are optimised for
- A modern GPU (RTX 4090 / A10G / A100) has **10k–16k+ CUDA cores**, so 28.4M legs across 30 jobs is trivially parallel

---

## Sequential Correctness on GPU

The pipeline is naturally staged, and GPU kernels implicitly synchronise between launches:

1. **Pricing kernel** — all 28.4M legs are fully independent → massively parallel
2. **Instrument rollup kernel** — depends on all legs being priced first, but is itself a single `scatter_add` across 70k bins per job → parallel within each kernel
3. **Underlying rollup kernel** — depends on instrument rollup → simple reduction
4. **Portfolio rollup** — single sum across instrument totals

Each CUDA kernel guarantees all threads complete before the next kernel launches. The rollup always sees complete, consistent data — no race conditions, no synchronisation code needed.

---

## RAM / VRAM Requirements

| Approach | Memory Needed | Notes |
|---|---|---|
| CPU 1 job at a time | +1.4GB above baseline | Observed: 25.6GB → 27GB |
| CPU 30 jobs parallel | +42GB above baseline (~68GB total) | 30 × 1.4GB worker overhead |
| CPU 6 × 5 batched | +7GB above baseline (~33GB total) | Sweet spot for 64GB machines |
| GPU 1 job at a time | ~1.5GB VRAM | Leg arrays + result buffers |
| GPU mega-batch (30 jobs) | ~4GB VRAM | 28.4M legs × ~150 bytes (input+output arrays) |

GPU VRAM is separate from system RAM — system memory stays untouched during pricing. A 24GB A10G handles the full 30-job mega-batch with 20GB to spare.

---

## Cloud Cost Comparison (AWS, On-Demand)

| Instance | Specs | $/hr | Time for 30 Jobs | Cost per Run |
|---|---|---|---|---|
| **c7i.4xlarge** (CPU) | 16 vCPU, 32GB | $0.71 | ~3.5 min (sequential) | **$0.041** |
| **c7i.8xlarge** (CPU) | 32 vCPU, 64GB | $1.43 | ~42s (6 batches of 5) | **$0.017** |
| **c7i.24xlarge** (CPU) | 96 vCPU, 192GB | $4.28 | ~12s (all 30 parallel) | **$0.014** |
| **g5.xlarge** (GPU) | 1× A10G 24GB, 4 vCPU, 16GB | $1.01 | ~2s (mega-batch) | **$0.0006** |
| **g5.2xlarge** (GPU) | 1× A10G 24GB, 8 vCPU, 32GB | $1.21 | ~2s (mega-batch) | **$0.0007** |
| **p4d.24xlarge** (GPU) | 8× A100 80GB, 96 vCPU, 1.1TB | $32.77 | ~1s | **$0.009** |

### Spot Pricing (Typical)

| Instance | On-Demand $/hr | Spot $/hr | Spot Savings |
|---|---|---|---|
| c7i.8xlarge | $1.43 | ~$0.50 | 65% |
| g5.xlarge | $1.01 | ~$0.35 | 65% |

The GPU instance is cheaper per hour than the CPU instance even at spot pricing.

---

## Cost at Scale

| Frequency | CPU (c7i.8xlarge) | GPU (g5.xlarge) | Savings |
|---|---|---|---|
| 1 run/day | $0.017/day → **$6.20/yr** | $0.0006/day → **$0.22/yr** | 28× cheaper |
| 10 runs/day | **$62/yr** | **$2.19/yr** | 28× cheaper |
| 100 runs/day (intraday risk) | **$620/yr** | **$21.90/yr** | 28× cheaper |
| 1,000 runs/day (real-time) | **$6,205/yr** | **$219/yr** | 28× cheaper |

---

## Head-to-Head Summary

| Factor | CPU | GPU | Winner |
|---|---|---|---|
| **Cost per run** | $0.014–$0.041 | $0.0006–$0.0007 | **GPU by 20–60×** |
| **Wall-clock time** | 12s – 3.5 min | 1–3s | **GPU by 6–100×** |
| **RAM pressure** | 33–68GB system RAM | 4GB VRAM (system RAM free) | **GPU** |
| **Instance hourly rate** | $0.50–$4.28/hr | $0.35–$1.01/hr | **GPU** |
| **Instance startup time** | ~30s | ~45–60s (driver init) | CPU slightly |
| **Code change required** | None (current code) | CuPy swap (~20 lines) | CPU slightly |
| **Spot instance availability** | High | Moderate | CPU slightly |

---

## Implementation Effort

Migrating the current CPU code to GPU is minimal:

1. Replace `import numpy as np` with `import cupy as cp`
2. Remove `ProcessPoolExecutor` — the GPU IS the parallel engine
3. Arrays created with `cp.array()` instead of `np.array()`
4. `scipy.stats.norm.cdf/pdf` → CuPy custom kernel or `cupyx.scipy.special`
5. Rollup `np.add.at` → `cp.add.at` (identical API)
6. Transfer final results back to CPU with `.get()` for JSON serialisation

Estimated diff: ~20 lines changed, same algorithmic structure.

---

## Recommendation

| Use Case | Best Choice | Why |
|---|---|---|
| **EOD batch (1 run/day)** | CPU (c7i.4xlarge) | Both cost nothing; CPU avoids GPU driver complexity |
| **Multiple daily runs (5–50/day)** | GPU (g5.xlarge) | 28× cheaper, 10× faster, frees system RAM |
| **Intraday risk (100+/day)** | GPU (g5.xlarge) | $22/yr vs $620/yr; 2-second latency enables near-real-time dashboards |
| **Real-time streaming** | GPU (g5.xlarge) | Only viable option at sub-3s per cycle |

The **g5.xlarge** (single A10G, $1.01/hr on-demand, ~$0.35/hr spot) is the sweet spot. It handles the full 30-job workload in 2 seconds with 4GB of its 24GB VRAM, costs $0.0006 per run, and the code migration is ~20 lines. The A100 (`p4d.24xlarge`) at $32.77/hr is unjustifiable unless the workload scales to 100x this size.
