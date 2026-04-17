#!/usr/bin/env python3
"""
=============================================================================
  PARALLEL PRICING SIMULATOR
=============================================================================
  70,000 structured instruments  |  400 active underlyings  |  ~945k legs
  PUT / FUNDING / COUPON legs    |  Delta, Gamma & P&L      |  Full rollup

  Pipeline:
    Layer 0  – mock 500 underlying market data (same pricing date)
    Layer 1  – parallel vectorised leg pricing (NumPy + ProcessPool)
    Layer 2  – instrument rollup  (leg → instrument)
    Layer 3  – portfolio rollup   (instrument → underlying → book)

  RAM is monitored continuously; when usage exceeds a threshold the
  in-memory result buffer is flushed to JSON intermediate files that
  subsequent layers read back transparently.

  A Rich live terminal dashboard shows worker activity, throughput,
  memory pressure, offload events and running portfolio Greeks/P&L.
=============================================================================
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import os
import sys
import json
import time
import shutil
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── third-party ──────────────────────────────────────────────────────────────
try:
    import numpy as np
    from scipy.stats import norm as sp_norm
    import psutil
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
except ImportError as exc:
    print(f"Missing dependency: {exc.name}")
    print("Install with:  pip install -r requirements.txt")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

NUM_INSTRUMENTS       = 70_000
NUM_MOCK_UNDERLYINGS  = 500          # total mocked market-data entries
NUM_ACTIVE_UNDERLYINGS = 400         # actually referenced by instruments
UL_PER_INSTRUMENT     = (4, 5)       # min / max underlyings per instrument
BUMP_PCT              = 0.01         # 1 % spot bump for sensi / P&L
NUM_WORKERS           = max(1, (os.cpu_count() or 8) - 1)
RAM_THRESHOLD_PCT     = 75.0         # flush to JSON above this %
FORCE_OFFLOAD_EVERY   = 100          # also force-flush every N underlyings
OFFLOAD_DIR           = "pricing_intermediates"
PRICING_DATE          = "2026-04-17"

# Leg-type codes
LEG_PUT     = 0
LEG_FUNDING = 1
LEG_COUPON  = 2
LEG_NAMES   = {0: "PUT", 1: "FUNDING", 2: "COUPON"}


# ═════════════════════════════════════════════════════════════════════════════
#  TICKER & MARKET DATA GENERATION
# ═════════════════════════════════════════════════════════════════════════════

_SEED_TICKERS = [
    "AAPL","MSFT","GOOG","AMZN","TSLA","META","NVDA","NFLX","AMD","INTC",
    "CRM","ORCL","CSCO","ADBE","PYPL","QCOM","TXN","AVGO","NOW","SNOW",
    "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","USB",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","TMO","ABT","BMY","AMGN",
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","HAL",
    "PG","KO","PEP","WMT","COST","HD","MCD","NKE","SBUX","TGT",
    "CAT","DE","HON","UPS","BA","GE","RTX","LMT","MMM","EMR",
    "NEE","DUK","SO","AEP","EXC","SRE","XEL","WEC","DTE","AES",
    "LIN","APD","ECL","SHW","DD","NEM","FCX","NUE","VMC","MLM",
    "DIS","CMCSA","VZ","TMUS","CHTR","EA","TTWO","RBLX","WBD","PARA",
]


def generate_tickers(n: int) -> list[str]:
    """Return *n* unique ticker symbols."""
    tickers = list(_SEED_TICKERS[: min(n, len(_SEED_TICKERS))])
    sectors = ["TEC", "FIN", "HLT", "ENR", "CON", "IND", "UTL", "MTL", "COM", "REL"]
    idx = 0
    while len(tickers) < n:
        sec = sectors[idx % len(sectors)]
        tickers.append(f"{sec}{idx // len(sectors):03d}")
        idx += 1
    return tickers[:n]


def generate_market_data(tickers: list[str], rng: np.random.Generator) -> dict:
    """Mock spot / vol / rate / div-yield for each underlying on PRICING_DATE."""
    return {
        t: {
            "spot":      round(float(rng.uniform(20.0, 500.0)), 2),
            "vol":       round(float(rng.uniform(0.15, 0.65)), 4),
            "rate":      round(float(rng.uniform(0.01, 0.055)), 4),
            "div_yield": round(float(rng.uniform(0.0, 0.04)), 4),
        }
        for t in tickers
    }


# ═════════════════════════════════════════════════════════════════════════════
#  INSTRUMENT & LEG GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_instruments_and_legs(
    n_instruments: int,
    tickers: list[str],
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    """
    Build 70 000 instruments, each referencing 4-5 underlyings, each with
    PUT + FUNDING + COUPON legs.  Returns
        legs_by_underlying : dict[ticker] → {instrument_ids, leg_types, …}
        instrument_meta    : list of per-instrument summaries
    """
    legs_by_underlying: dict = defaultdict(lambda: {
        "instrument_ids": [], "leg_types": [], "strikes": [],
        "notionals": [], "maturities": [],
    })
    instrument_meta: list[dict] = []
    n_tickers = len(tickers)

    for inst_id in range(n_instruments):
        n_ul = int(rng.integers(UL_PER_INSTRUMENT[0], UL_PER_INSTRUMENT[1] + 1))
        chosen = rng.choice(n_tickers, size=n_ul, replace=False)
        chosen_tickers = [tickers[i] for i in chosen]

        for ticker in chosen_tickers:
            bucket = legs_by_underlying[ticker]
            # ~50 % long, ~50 % short to create realistic netting
            sign = 1.0 if rng.random() < 0.5 else -1.0
            for leg_type in (LEG_PUT, LEG_FUNDING, LEG_COUPON):
                bucket["instrument_ids"].append(inst_id)
                bucket["leg_types"].append(leg_type)
                bucket["strikes"].append(
                    round(float(rng.uniform(0.80, 1.20)), 4)
                    if leg_type == LEG_PUT else 1.0
                )
                bucket["notionals"].append(round(float(sign * rng.uniform(1_000, 50_000)), 2))
                bucket["maturities"].append(round(float(rng.uniform(0.25, 5.0)), 2))

        instrument_meta.append({
            "id": inst_id,
            "underlyings": chosen_tickers,
            "n_legs": len(chosen_tickers) * 3,
        })

    return dict(legs_by_underlying), instrument_meta


# ═════════════════════════════════════════════════════════════════════════════
#  PRICING ENGINE  (module-level functions — must be picklable for Windows)
# ═════════════════════════════════════════════════════════════════════════════

def _bs_put_vectors(S, K, T, r, q, sigma):
    """Vectorised Black-Scholes put: returns (price, delta, gamma)."""
    T = np.maximum(T, 1e-10)
    sqrt_T   = np.sqrt(T)
    sig_sqrt = sigma * sqrt_T
    sig_sqrt = np.maximum(sig_sqrt, 1e-10)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sig_sqrt
    d2 = d1 - sig_sqrt

    eq_disc = np.exp(-q * T)
    rt_disc = np.exp(-r * T)

    Nd1n = sp_norm.cdf(-d1)
    Nd2n = sp_norm.cdf(-d2)
    nd1  = sp_norm.pdf(d1)

    price = K * rt_disc * Nd2n - S * eq_disc * Nd1n
    delta = -Nd1n * eq_disc
    gamma = nd1 * eq_disc / (S * sig_sqrt)
    return price, delta, gamma


def price_underlying_task(task: dict) -> dict:
    """
    Price every leg for a single underlying.  Runs in a worker process.
    Heavy lifting is fully vectorised via NumPy.
    """
    ticker    = task["ticker"]
    spot      = task["spot"]
    vol       = task["vol"]
    rate      = task["rate"]
    div_yield = task["div_yield"]
    bump_pct  = task["bump_pct"]
    legs      = task["legs"]

    inst_ids    = np.array(legs["instrument_ids"], dtype=np.int32)
    leg_types   = np.array(legs["leg_types"],      dtype=np.int8)
    strike_pcts = np.array(legs["strikes"],        dtype=np.float64)
    notionals   = np.array(legs["notionals"],      dtype=np.float64)
    maturities  = np.array(legs["maturities"],     dtype=np.float64)

    n       = len(inst_ids)
    strikes = strike_pcts * spot
    bump    = spot * bump_pct

    prices = np.zeros(n)
    deltas = np.zeros(n)
    gammas = np.zeros(n)
    pls    = np.zeros(n)

    # ── PUT legs ─────────────────────────────────────────────────────────────
    m = leg_types == LEG_PUT
    if m.any():
        K = strikes[m]; T = maturities[m]; N = notionals[m]
        p, d, g     = _bs_put_vectors(spot,        K, T, rate, div_yield, vol)
        p_up, _, _  = _bs_put_vectors(spot + bump, K, T, rate, div_yield, vol)
        prices[m] = p * N
        deltas[m] = d * N
        gammas[m] = g * N
        pls[m]    = (p_up - p) * N

    # ── FUNDING legs ─────────────────────────────────────────────────────────
    m = leg_types == LEG_FUNDING
    if m.any():
        T = maturities[m]; N = notionals[m]
        disc = np.exp(-rate * T)
        prices[m] = N * (1.0 - disc)
        deltas[m] = N * disc * 0.005        # rate-sensitivity proxy
        gammas[m] = 0.0
        pls[m]    = deltas[m] * bump

    # ── COUPON legs ──────────────────────────────────────────────────────────
    m = leg_types == LEG_COUPON
    if m.any():
        T = maturities[m]; N = notionals[m]
        disc = np.exp(-rate * T)
        coupon = 0.05
        prices[m] = N * coupon * T * disc
        deltas[m] = N * coupon * T * 0.001   # small equity sensitivity
        gammas[m] = 0.0
        pls[m]    = deltas[m] * bump

    return {
        "ticker":         ticker,
        "n_legs":         n,
        "instrument_ids": inst_ids.tolist(),
        "deltas":         deltas.tolist(),
        "gammas":         gammas.tolist(),
        "prices":         prices.tolist(),
        "pls":            pls.tolist(),
        "total_delta":    float(deltas.sum()),
        "total_gamma":    float(gammas.sum()),
        "total_price":    float(prices.sum()),
        "total_pl":       float(pls.sum()),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  RAM MONITOR & JSON OFFLOADING
# ═════════════════════════════════════════════════════════════════════════════

def ram_snapshot() -> tuple[float, float, float]:
    """Return (pct_used, used_gb, total_gb)."""
    mem = psutil.virtual_memory()
    return mem.percent, mem.used / (1024 ** 3), mem.total / (1024 ** 3)


def offload_to_json(buf: dict, directory: str, batch_num: int) -> tuple[str, int]:
    """Flush *buf* to a numbered JSON file; return (path, size_bytes)."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"layer1_batch_{batch_num:04d}.json")
    with open(path, "w") as fh:
        json.dump(buf, fh)
    return path, os.path.getsize(path)


def load_json(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


# ═════════════════════════════════════════════════════════════════════════════
#  ROLLUP CALCULATIONS
# ═════════════════════════════════════════════════════════════════════════════

def rollup_to_instruments(all_results: list[dict], n_instruments: int) -> dict:
    """Leg-level → instrument-level aggregation (uses np.add.at)."""
    d = np.zeros(n_instruments)
    g = np.zeros(n_instruments)
    p = np.zeros(n_instruments)
    pl = np.zeros(n_instruments)
    lc = np.zeros(n_instruments, dtype=np.int32)

    for r in all_results:
        ids = np.array(r["instrument_ids"], dtype=np.int32)
        np.add.at(d,  ids, np.array(r["deltas"]))
        np.add.at(g,  ids, np.array(r["gammas"]))
        np.add.at(p,  ids, np.array(r["prices"]))
        np.add.at(pl, ids, np.array(r["pls"]))
        np.add.at(lc, ids, 1)

    return dict(delta=d, gamma=g, price=p, pl=pl, leg_count=lc)


def rollup_to_underlyings(all_results: list[dict]) -> dict:
    """Per-underlying totals."""
    return {
        r["ticker"]: {
            "delta": r["total_delta"], "gamma": r["total_gamma"],
            "price": r["total_price"], "pl":    r["total_pl"],
            "n_legs": r["n_legs"],
        }
        for r in all_results
    }


def rollup_portfolio(inst: dict) -> dict:
    """Instrument-level → portfolio-level."""
    return {
        "total_delta": float(inst["delta"].sum()),
        "total_gamma": float(inst["gamma"].sum()),
        "total_price": float(inst["price"].sum()),
        "total_pl":    float(inst["pl"].sum()),
        "n_instruments": len(inst["delta"]),
        "total_legs":    int(inst["leg_count"].sum()),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

class DashState:
    """Mutable dashboard counters — updated by the main thread only."""

    def __init__(self, n_ul, n_inst, n_workers, total_legs):
        self.n_ul         = n_ul
        self.n_inst       = n_inst
        self.n_workers    = n_workers
        self.total_legs   = total_legs
        self.done_ul      = 0
        self.done_legs    = 0
        self.start        = time.time()
        self.phase        = "Initialising"
        self.ram_pct      = 0.0
        self.ram_used     = 0.0
        self.ram_total    = 0.0
        self.peak_ram_pct = 0.0
        self.offloads     = 0
        self.offload_mb   = 0.0
        self.in_mem_cnt   = 0
        self.run_delta    = 0.0
        self.run_gamma    = 0.0
        self.run_pl       = 0.0
        self.recent: list[str] = []
        self.active       = 0
        self.rollup_msg   = ""

    def elapsed(self):
        return time.time() - self.start

    def refresh_ram(self):
        self.ram_pct, self.ram_used, self.ram_total = ram_snapshot()
        if self.ram_pct > self.peak_ram_pct:
            self.peak_ram_pct = self.ram_pct


def _bar(pct: float, width: int = 40) -> str:
    filled = int(width * min(pct, 100) / 100)
    return "█" * filled + "░" * (width - filled)


def _signed(v: float) -> tuple[str, str]:
    s = f"{'+' if v >= 0 else ''}{v:,.2f}"
    return s, ("green" if v >= 0 else "red")


def build_dashboard(st: DashState) -> Panel:
    """Compose the Rich renderable for one frame."""
    e = st.elapsed()
    lps = st.done_legs / max(e, 0.001)
    eta = (st.total_legs - st.done_legs) / max(lps, 1)
    pct = st.done_ul / max(st.n_ul, 1) * 100

    # ── pipeline ribbon ──────────────────────────────────────────────────
    stages = {
        "Generate":  "✓" if st.phase != "Initialising" else "▶",
        "Pricing":   f"{pct:.0f}%" if "Pricing" in st.phase else ("✓" if st.done_ul == st.n_ul else "○"),
        "Rollup":    "▶" if "Rollup" in st.phase else ("✓" if "Complete" in st.phase else "○"),
        "Summary":   "✓" if "Complete" in st.phase else "○",
    }
    ribbon = Text("  ")
    for i, (name, mark) in enumerate(stages.items()):
        style = "bold green" if mark == "✓" else ("bold yellow" if mark not in ("○",) else "dim")
        ribbon.append(f"[{mark} {name}]", style=style)
        if i < len(stages) - 1:
            ribbon.append(" ══▶ ", style="dim")

    # ── progress ─────────────────────────────────────────────────────────
    prog = Text()
    prog.append(f"  Underlyings  [{_bar(pct)}] ", style="green")
    prog.append(f"{st.done_ul}/{st.n_ul} ({pct:.1f}%)\n", style="bold green")
    prog.append(f"  Legs priced  {st.done_legs:>12,} / {st.total_legs:,}\n", style="cyan")
    prog.append(f"  In flight    {st.active:>12,} futures", style="yellow")

    # ── timing ───────────────────────────────────────────────────────────
    timing = Text()
    timing.append(f"  Elapsed {e:>7.1f}s", style="dim")
    timing.append(f"  │  {lps:>11,.0f} legs/sec", style="bold magenta")
    timing.append(f"  │  ETA {eta:>6.1f}s", style="dim")

    # ── memory ───────────────────────────────────────────────────────────
    ram_style = "red" if st.ram_pct > RAM_THRESHOLD_PCT else "green"
    mem = Text()
    mem.append(f"  RAM  [{_bar(st.ram_pct, 24)}] ", style=ram_style)
    mem.append(f"{st.ram_used:.1f} / {st.ram_total:.1f} GB ({st.ram_pct:.0f}%)\n", style=ram_style)
    mem.append(f"  Peak RAM {st.peak_ram_pct:.0f}%", style="dim")
    mem.append(f"  │  JSON offloads {st.offloads}", style="dim")
    mem.append(f"  │  On disk {st.offload_mb:.1f} MB", style="dim")
    mem.append(f"  │  In memory {st.in_mem_cnt} results", style="dim")

    # ── workers ──────────────────────────────────────────────────────────
    active_cnt = min(st.active, st.n_workers)
    wk = Text()
    wk.append(f"  Active [{_bar(active_cnt / max(st.n_workers, 1) * 100, 16)}] ", style="blue")
    wk.append(f"{active_cnt}/{st.n_workers}\n", style="bold blue")
    wk.append("  Recent: ", style="dim")
    for t in st.recent[-12:]:
        wk.append(f"{t} ", style="bold blue")

    # ── running totals ───────────────────────────────────────────────────
    ds, dc = _signed(st.run_delta)
    gs, gc = _signed(st.run_gamma)
    ps, pc = _signed(st.run_pl)
    tots = Text()
    tots.append(f"  Σ Delta  {ds:>22}\n", style=f"bold {dc}")
    tots.append(f"  Σ Gamma  {gs:>22}\n", style=f"bold {gc}")
    tots.append(f"  Σ P&L    {ps:>22}",   style=f"bold {pc}")

    parts: list = [
        ribbon, Text(""),
        Panel(prog,    title="Progress",  border_style="blue",    box=box.ROUNDED),
        Panel(timing,  title="Timing",    border_style="magenta", box=box.ROUNDED),
        Panel(mem,     title="Memory",    border_style="yellow",  box=box.ROUNDED),
        Panel(wk,      title="Workers",   border_style="blue",    box=box.ROUNDED),
        Panel(tots,    title="Running Totals", border_style="green", box=box.ROUNDED),
    ]
    if st.rollup_msg:
        parts.append(Panel(
            Text(f"  {st.rollup_msg}", style="cyan"),
            title="Rollup", border_style="cyan", box=box.ROUNDED,
        ))

    return Panel(
        Group(*parts),
        title=f"[bold white] PARALLEL PRICING ENGINE — {PRICING_DATE} [/bold white]",
        border_style="bright_white", box=box.DOUBLE,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def main():
    console = Console()

    # House-keeping
    if os.path.exists(OFFLOAD_DIR):
        shutil.rmtree(OFFLOAD_DIR)
    os.makedirs(OFFLOAD_DIR, exist_ok=True)

    rng = np.random.default_rng(42)

    # ── Phase 1 : data generation ────────────────────────────────────────
    console.print("\n[bold cyan]Phase 1[/bold cyan]  Generating tickers, market data & instruments …")
    t0 = time.time()

    all_tickers   = generate_tickers(NUM_MOCK_UNDERLYINGS)        # 500 mocked
    market_data   = generate_market_data(all_tickers, rng)        # all 500

    active_tickers = all_tickers[:NUM_ACTIVE_UNDERLYINGS]         # 400 used
    legs_by_ul, inst_meta = generate_instruments_and_legs(
        NUM_INSTRUMENTS, active_tickers, rng,
    )

    total_legs = sum(len(v["instrument_ids"]) for v in legs_by_ul.values())
    gen_time   = time.time() - t0

    # Save Layer 0 : market data
    layer0_path = os.path.join(OFFLOAD_DIR, "layer0_market_data.json")
    with open(layer0_path, "w") as fh:
        json.dump({"date": PRICING_DATE, "data": market_data}, fh)

    console.print(f"  Mock underlyings : {NUM_MOCK_UNDERLYINGS:,}")
    console.print(f"  Active tickers   : {len(legs_by_ul):,}")
    console.print(f"  Instruments      : {NUM_INSTRUMENTS:,}")
    console.print(f"  Total legs       : {total_legs:,}")
    console.print(f"  Generated in     : {gen_time:.2f}s\n")

    # ── Build task list (largest-first for load balance) ─────────────────
    tasks = []
    for ticker, legs in legs_by_ul.items():
        md = market_data[ticker]
        tasks.append({
            "ticker":    ticker,
            "spot":      md["spot"],
            "vol":       md["vol"],
            "rate":      md["rate"],
            "div_yield": md["div_yield"],
            "bump_pct":  BUMP_PCT,
            "legs":      legs,
        })
    tasks.sort(key=lambda t: len(t["legs"]["instrument_ids"]), reverse=True)

    # Free the large generation dict – data lives in `tasks` now
    del legs_by_ul

    # ── Phase 2 : parallel pricing ───────────────────────────────────────
    st = DashState(len(tasks), NUM_INSTRUMENTS, NUM_WORKERS, total_legs)
    st.phase = "Pricing"
    st.refresh_ram()

    mem_results: dict[str, dict]   = {}   # results kept in RAM
    disk_paths:  list[str]         = []   # JSON paths for offloaded batches
    buf:         dict[str, dict]   = {}   # buffer awaiting flush decision
    offload_seq = 0

    with Live(build_dashboard(st), refresh_per_second=12, console=console) as live:

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futures = {
                pool.submit(price_underlying_task, t): t["ticker"]
                for t in tasks
            }
            st.active = len(futures)
            live.update(build_dashboard(st))

            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    console.print(f"[red]  ✗ {ticker}: {exc}[/red]")
                    st.active -= 1
                    continue

                # Accumulate
                st.done_ul   += 1
                st.done_legs += result["n_legs"]
                st.active    -= 1
                st.run_delta += result["total_delta"]
                st.run_gamma += result["total_gamma"]
                st.run_pl    += result["total_pl"]
                st.recent.append(ticker)
                buf[ticker]   = result

                # ── RAM gate / periodic flush ────────────────────────────
                if st.done_ul % FORCE_OFFLOAD_EVERY == 0 or st.done_ul == st.n_ul:
                    st.refresh_ram()
                    need_flush = (
                        st.ram_pct > RAM_THRESHOLD_PCT
                        or st.done_ul % FORCE_OFFLOAD_EVERY == 0
                    )
                    if need_flush and buf:
                        path, sz = offload_to_json(buf, OFFLOAD_DIR, offload_seq)
                        disk_paths.append(path)
                        st.offloads   += 1
                        st.offload_mb += sz / (1024 ** 2)
                        offload_seq   += 1
                        buf = {}
                    elif buf:
                        mem_results.update(buf)
                        st.in_mem_cnt += len(buf)
                        buf = {}

                live.update(build_dashboard(st))

        # Flush anything remaining
        if buf:
            st.refresh_ram()
            if st.ram_pct > RAM_THRESHOLD_PCT:
                path, sz = offload_to_json(buf, OFFLOAD_DIR, offload_seq)
                disk_paths.append(path)
                st.offloads   += 1
                st.offload_mb += sz / (1024 ** 2)
            else:
                mem_results.update(buf)
                st.in_mem_cnt += len(buf)
            buf = {}

        pricing_time = st.elapsed()
        st.refresh_ram()
        live.update(build_dashboard(st))

        # ── Phase 3 : rollup ─────────────────────────────────────────────
        st.phase = "Rollup"
        st.rollup_msg = "Collecting results from memory + disk …"
        live.update(build_dashboard(st))

        all_results: list[dict] = list(mem_results.values())
        for p in disk_paths:
            batch = load_json(p)
            all_results.extend(batch.values())

        # Free in-memory buffer now that we have the unified list
        mem_results.clear()

        st.rollup_msg = f"Instrument rollup ({NUM_INSTRUMENTS:,} instruments) …"
        live.update(build_dashboard(st))
        t_roll = time.time()

        inst_roll = rollup_to_instruments(all_results, NUM_INSTRUMENTS)

        st.rollup_msg = "Underlying rollup …"
        live.update(build_dashboard(st))
        ul_roll = rollup_to_underlyings(all_results)

        st.rollup_msg = "Portfolio rollup …"
        live.update(build_dashboard(st))
        port = rollup_portfolio(inst_roll)
        rollup_time = time.time() - t_roll

        # ── Save Layer 2 : instrument rollup ─────────────────────────────
        st.rollup_msg = "Writing Layer 2 (instrument rollup) …"
        live.update(build_dashboard(st))
        l2_path = os.path.join(OFFLOAD_DIR, "layer2_instrument_rollup.json")
        with open(l2_path, "w") as fh:
            json.dump({k: v.tolist() if hasattr(v, "tolist") else v
                       for k, v in inst_roll.items()}, fh)

        # ── Save Layer 3 : portfolio summary ─────────────────────────────
        st.rollup_msg = "Writing Layer 3 (portfolio summary) …"
        live.update(build_dashboard(st))
        l3_path = os.path.join(OFFLOAD_DIR, "layer3_portfolio_summary.json")
        port.update({
            "pricing_date":     PRICING_DATE,
            "pricing_time_sec": round(pricing_time, 3),
            "rollup_time_sec":  round(rollup_time, 3),
            "total_time_sec":   round(pricing_time + rollup_time, 3),
        })
        with open(l3_path, "w") as fh:
            json.dump(port, fh, indent=2)

        st.phase = "Complete"
        st.rollup_msg = f"All layers written — rollup in {rollup_time:.2f}s"
        live.update(build_dashboard(st))
        time.sleep(1.5)   # hold final frame

    # ═════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    pl_s, pl_c = _signed(port["total_pl"])

    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold]Pricing Date :[/bold] {PRICING_DATE}\n"
            f"[bold]Instruments  :[/bold] {port['n_instruments']:>10,}\n"
            f"[bold]Total Legs   :[/bold] {port['total_legs']:>10,}\n"
            f"[bold]Underlyings  :[/bold] {len(ul_roll):>10,}\n"
            f"\n"
            f"[bold]Σ Delta :[/bold] {port['total_delta']:>22,.2f}\n"
            f"[bold]Σ Gamma :[/bold] {port['total_gamma']:>22,.2f}\n"
            f"[bold]Σ Price :[/bold] {port['total_price']:>22,.2f}\n"
            f"[bold]Σ P&L   :[/bold] [{pl_c}]{pl_s:>22}[/{pl_c}]\n"
            f"\n"
            f"[dim]Pricing   {pricing_time:>8.2f}s[/dim]\n"
            f"[dim]Rollup    {rollup_time:>8.2f}s[/dim]\n"
            f"[dim]Total     {pricing_time + rollup_time:>8.2f}s[/dim]\n"
            f"[dim]Peak RAM  {st.peak_ram_pct:.0f}%   │   "
            f"Offloads {st.offloads}  ({st.offload_mb:.1f} MB)   │   "
            f"Workers {NUM_WORKERS}[/dim]\n"
            f"\n"
            f"[dim]Intermediate dir: {OFFLOAD_DIR}/[/dim]"
        ),
        title="[bold green] PRICING COMPLETE [/bold green]",
        border_style="green", box=box.DOUBLE,
    ))

    # ── Top 10 instruments by |P&L| ─────────────────────────────────────
    top_n   = 10
    abs_pl  = np.abs(inst_roll["pl"])
    top_idx = np.argsort(abs_pl)[-top_n:][::-1]

    tbl = Table(title="Top 10 Instruments by |P&L|", box=box.ROUNDED)
    tbl.add_column("Inst ID", style="cyan",    justify="right")
    tbl.add_column("Delta",                     justify="right")
    tbl.add_column("Gamma",                     justify="right")
    tbl.add_column("Price",                     justify="right")
    tbl.add_column("P&L",                       justify="right")
    tbl.add_column("Legs",   style="dim",       justify="right")

    for i in top_idx:
        s, c = _signed(inst_roll["pl"][i])
        tbl.add_row(
            f"{i:,}",
            f"{inst_roll['delta'][i]:,.2f}",
            f"{inst_roll['gamma'][i]:,.2f}",
            f"{inst_roll['price'][i]:,.2f}",
            f"[{c}]{s}[/{c}]",
            f"{inst_roll['leg_count'][i]:,}",
        )
    console.print(tbl)

    # ── Top 10 underlyings by |delta| ───────────────────────────────────
    ul_tbl = Table(title="Top 10 Underlyings by |Delta|", box=box.ROUNDED)
    ul_tbl.add_column("Ticker", style="cyan")
    ul_tbl.add_column("Delta",  justify="right")
    ul_tbl.add_column("Gamma",  justify="right")
    ul_tbl.add_column("P&L",   justify="right")
    ul_tbl.add_column("Legs",  style="dim", justify="right")

    sorted_ul = sorted(ul_roll.items(), key=lambda x: abs(x[1]["delta"]), reverse=True)[:top_n]
    for tk, v in sorted_ul:
        s, c = _signed(v["pl"])
        ul_tbl.add_row(tk, f"{v['delta']:,.2f}", f"{v['gamma']:,.2f}",
                        f"[{c}]{s}[/{c}]", f"{v['n_legs']:,}")
    console.print(ul_tbl)

    # ── Intermediate files listing ───────────────────────────────────────
    ftbl = Table(title="Intermediate Layer Files", box=box.ROUNDED)
    ftbl.add_column("File",  style="cyan")
    ftbl.add_column("Size",  justify="right")
    for root, _, files in os.walk(OFFLOAD_DIR):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            sz = os.path.getsize(fp)
            if sz > 1024 * 1024:
                ftbl.add_row(fn, f"{sz / (1024**2):.1f} MB")
            else:
                ftbl.add_row(fn, f"{sz / 1024:.0f} KB")
    console.print(ftbl)

    # ── Taylor P&L validation (spot-check) ───────────────────────────────
    console.print()
    console.print("[dim]Taylor P&L validation (portfolio level):[/dim]")
    taylor_pl = port["total_delta"] * (BUMP_PCT * 100) + 0.5 * port["total_gamma"] * (BUMP_PCT * 100) ** 2
    console.print(f"  [dim]Σ(δ·ΔS + ½γ·ΔS²) ≈ {taylor_pl:>18,.2f}[/dim]")
    console.print(f"  [dim]Actual Σ P&L       = {port['total_pl']:>18,.2f}[/dim]")
    console.print()


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
