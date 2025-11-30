#!/usr/bin/env python3
import os, csv, json, time, math
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone
from constants import (
    OUT_DIR, 
    BIN_LAG_SEC, 
    FLUSH_EVERY_SEC, 
    AGGREGATOR_SLEEP_S,
    LATENCY_CSV,
    SYSTEM_SAMPLE_CSV,
    PERF_CSV,
    SUMMARY_1S_CSV,
    AGGREGATOR_STATE_FILE,
)

METRICS_DIR = Path(OUT_DIR)
INF_FILE    = Path(LATENCY_CSV)
SYS_FILE    = Path(SYSTEM_SAMPLE_CSV)
PERF_FILE   = Path(PERF_CSV)  # may not exist
STATE_FILE  = Path(AGGREGATOR_STATE_FILE)
OUT_FILE    = Path(SUMMARY_1S_CSV)


# ---- helpers ----
def iso_to_epoch_ms(s: str) -> int:
    # s = "2025-10-13T17:20:01.123Z" or ISO without Z
    dt = datetime.fromisoformat(s.replace("Z","+00:00"))
    return int(dt.timestamp() * 1000)


def floor_to_second(epoch_ms: int) -> int:
    return (epoch_ms // 1000) * 1000


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"offsets": {}, "last_flushed_ms": 0}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


def ensure_out_header():
    if not OUT_FILE.exists() or OUT_FILE.stat().st_size == 0:
        with OUT_FILE.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "Time","ts_unix_ms","city","model","context_hours","horizon_hours",
                "rps","p50","p90","p99",
                "cpu_proc_pct","proc_rss_mb",
                "cpu_total_pct","mem_used_mb",
                "ipc","miss_rate","llc_mpki"
            ])


# ---- parsers for each CSV ----
def tail_lines(path: Path, state, key):
    """Yield new CSV rows (as lists) since last offset; update offset in state."""
    
    offsets = state["offsets"]
    off = offsets.get(key, 0)
    if not path.exists():
        return []
    
    size = path.stat().st_size
    if off > size:  # file rotated or truncated
        off = 0
    
    out = []
    with path.open("r", newline="") as f:
        f.seek(off)
        rdr = csv.reader(f)
        # if starting at 0, skip header
        if off == 0:
            try:
                next(rdr)
            except StopIteration:
                pass
    
        for row in rdr:
            out.append(row)
    
        offsets[key] = f.tell()
    
    return out


# in-memory bins: dict[(sec_ms, city, model, ctx, hor)] -> accumulators
class Bin:
    __slots__ = ("lats","proc_cpu_sum","proc_cpu_cnt","rss_sum","rss_cnt",
                 "cpu_total_sum","cpu_total_cnt","mem_used_sum","mem_used_cnt",
                 "instructions","cycles","cache_miss","cache_ref","llc_miss","count")
    
    def __init__(self):
        self.lats = []  # store latencies for quantiles
        self.proc_cpu_sum = self.proc_cpu_cnt = 0.0
        self.rss_sum = self.rss_cnt = 0.0
        self.cpu_total_sum = self.cpu_total_cnt = 0.0
        self.mem_used_sum = self.mem_used_cnt = 0.0
        self.instructions = 0.0
        self.cycles = 0.0
        self.cache_miss = 0.0
        self.cache_ref = 0.0
        self.llc_miss = 0.0
        self.count = 0

    def add_lat(self, v): 
        self.lats.append(float(v)); self.count += 1
    
    def add_proc(self, cpu, rss):
        self.proc_cpu_sum += cpu; self.proc_cpu_cnt += 1
        self.rss_sum += rss; self.rss_cnt += 1
    
    def add_sys(self, cpu_total, mem_used):
        self.cpu_total_sum += cpu_total; self.cpu_total_cnt += 1
        self.mem_used_sum += mem_used; self.mem_used_cnt += 1
    
    def add_perf(self, name, val):
        v = float(val)
        if name == "instructions": self.instructions += v
        elif name == "cycles": self.cycles += v
        elif name == "cache_misses": self.cache_miss += v
        elif name == "cache_references": self.cache_ref += v
        elif name == "llc_load_misses": self.llc_miss += v

    def finalize(self):
        # quantiles
        if self.lats:
            arr = sorted(self.lats)
            def q(p):  # p in [0,100]
                if not arr: return math.nan
                k = (len(arr)-1)*p/100.0
                f = math.floor(k); c = math.ceil(k)
                if f==c: return arr[int(k)]
                return arr[f] + (arr[c]-arr[f])*(k-f)
            p50, p90, p99 = q(50), q(90), q(99)
            rps = len(arr)  # per second bin, count = RPS
        else:
            p50 = p90 = p99 = math.nan; rps = 0
        
        cpu_proc = (self.proc_cpu_sum / self.proc_cpu_cnt) if self.proc_cpu_cnt else math.nan
        rss     = (self.rss_sum / self.rss_cnt) if self.rss_cnt else math.nan
        cpu_tot = (self.cpu_total_sum / self.cpu_total_cnt) if self.cpu_total_cnt else math.nan
        mem_used= (self.mem_used_sum / self.mem_used_cnt) if self.mem_used_cnt else math.nan
        ipc = (self.instructions / self.cycles) if self.cycles else math.nan
        miss_rate = (self.cache_miss / self.cache_ref) if self.cache_ref else math.nan
        llc_mpki  = (self.llc_miss / (self.instructions/1000.0)) if self.instructions else math.nan
        
        return rps, p50, p90, p99, cpu_proc, rss, cpu_tot, mem_used, ipc, miss_rate, llc_mpki


bins = defaultdict(Bin)  # key -> Bin


def inf_key(sec_ms, city, model, ctx, hor): 
    return (sec_ms, city, model, int(ctx), int(hor))


def process_inference_rows(rows):
    # perf_inference.csv header:
    # ts,city,model,context_hours,horizon_hours,latency_s,cpu_proc_pct,proc_rss_mb
    for r in rows:
        if len(r) < 8: 
            continue
        
        ts, city, model, ctx, hor, lat, p_cpu, rss = r[:8]
        
        try:
            ms = floor_to_second(iso_to_epoch_ms(ts))
        except Exception:
            continue
        
        k = inf_key(ms, city, model, ctx, hor)
        b = bins[k]
        b.add_lat(lat)
        # also fold in the per-call proc cpu / rss (avg across bin)
        
        try: 
            b.add_proc(float(p_cpu), float(rss))
        except: 
            pass


def process_sys_rows(rows):
    # sys_sample.csv: ts,cpu_total_pct,cpu_user_pct,cpu_sys_pct,mem_used_mb,mem_avail_mb,proc_cpu_pct,proc_rss_mb
    for r in rows:
        if len(r) < 8: 
            continue
        
        ts, cpu_tot, _cpu_user, _cpu_sys, mem_used, _mem_avail, p_cpu, rss = r[:8]

        try:
            ms = floor_to_second(iso_to_epoch_ms(ts))
        except Exception:
            continue

        for (sec, city, model, ctx, hor), b in list(bins.items()):
            if sec == ms:
                try:
                    b.add_sys(float(cpu_tot), float(mem_used))
                    b.add_proc(float(p_cpu), float(rss))  # reinforce process stats at higher freq
                except: 
                    pass


# derive base epoch for perf stream (on first row seen)
PERF_EPOCH0_MS = None  # will be set on first perf row based on current wall clock

def perf_event_to_key(event: str):
    """
    Normalize perf event names from hybrid CPUs.
    Input like: 'cpu_core/cache-references/' or 'cpu_atom/LLC-load-misses/'
    Return canonical keys we aggregate on: 'cache_references','cache_misses','llc_load_misses','instructions','cycles'
    """
    e = event.strip().strip("/")  # 'cpu_core/cache-references'
    # drop the hybrid prefix if present
    parts = e.split("/", 1)
    if len(parts) == 2 and parts[0] in ("cpu_core", "cpu_atom"):
        e = parts[1]  # 'cache-references'
    # map to canonical
    mapping = {
        "cache-references": "cache_references",
        "cache-misses": "cache_misses",
        "LLC-load-misses": "llc_load_misses",
        "instructions": "instructions",
        "cycles": "cycles",
        # optional extras:
        "branches": "branches",
        "branch-misses": "branch_misses",
    }
    return mapping.get(e)


def process_perf_rows(rows):
    """
    Parse perf -I 1000 -x , rows like:
    <time_s>,<value>,<unit>,<event>,<enabled>,<running>,<ratio?>,<extra?>
    On hybrid CPUs, event looks like 'cpu_core/cache-misses/' or 'cpu_atom/instructions/'.
    We:
      - convert time_s -> epoch ms using a base captured at first row,
      - normalize event name and sum atom+core,
      - attribute counters to ALL active bins in that second (coarse but fine when one workload runs).
    """
    global PERF_EPOCH0_MS
    now_ms = int(time.time() * 1000)
    for r in rows:
        if len(r) < 4:
            continue
        try:
            t_s = float(r[0])        # seconds since perf start
            val = float(r[1])        # counter delta for this interval
            ev  = r[3].strip()       # event text
        except Exception:
            continue

        # lazily set base epoch so: epoch_ms = PERF_EPOCH0_MS + t_s*1000
        if PERF_EPOCH0_MS is None:
            # assume first perf row corresponds to 'now'; anchor back by t_s
            PERF_EPOCH0_MS = now_ms - int(t_s * 1000)

        metric = perf_event_to_key(ev)
        if metric is None:
            continue

        sec_ms = floor_to_second(PERF_EPOCH0_MS + int(t_s * 1000))

        # Sum across all active bins for that second (if you interleave workloads heavily,
        # switch to tracking "current label" and assign only to that bin).
        for (sec, city, model, ctx, hor), b in list(bins.items()):
            if sec == sec_ms:
                # our Bin.add_perf expected canonical names with underscores:
                if metric == "instructions":
                    b.instructions += val
                elif metric == "cycles":
                    b.cycles += val
                elif metric == "cache_misses":
                    b.cache_miss += val
                elif metric == "cache_references":
                    b.cache_ref += val
                elif metric == "llc_load_misses":
                    b.llc_miss += val
                # branches/branch_misses optional: you can store if you want


def flush_ready_bins(now_ms):
    """Flush bins whose second < now - BIN_LAG_SEC to OUT_FILE; remove them from memory."""
    cutoff = now_ms - BIN_LAG_SEC*1000
    if not bins:
        return 0
    
    rows = []
    for (sec_ms, city, model, ctx, hor), b in list(bins.items()):
        if sec_ms <= cutoff:
            rps, p50, p90, p99, cpu_proc, rss, cpu_tot, mem_used, ipc, miss_rate, llc = b.finalize()
            ts_iso = datetime.fromtimestamp(sec_ms/1000, tz=timezone.utc).isoformat()
            rows.append([ts_iso, sec_ms, city, model, ctx, hor,
                         rps, p50, p90, p99, cpu_proc, rss, cpu_tot, mem_used,
                         ipc, miss_rate, llc])
            del bins[(sec_ms, city, model, ctx, hor)]
    
    if rows:
        with OUT_FILE.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)
    
    return len(rows)


def main_loop():
    state = load_state()
    ensure_out_header()
    last_flush = 0
    print("stream_aggregate: running")

    while True:
        now_ms = int(time.time()*1000)

        inf_rows  = tail_lines(INF_FILE,  state, "inf")
        sys_rows  = tail_lines(SYS_FILE,  state, "sys")
        perf_rows = tail_lines(PERF_FILE, state, "perf") if PERF_FILE.exists() else []

        if inf_rows:  process_inference_rows(inf_rows)
        if sys_rows:  process_sys_rows(sys_rows)
        if perf_rows: process_perf_rows(perf_rows)

        if now_ms - last_flush >= FLUSH_EVERY_SEC*1000:
            flushed = flush_ready_bins(now_ms)
            save_state(state)
            last_flush = now_ms
            # print(f"flushed {flushed} rows")

        time.sleep(AGGREGATOR_SLEEP_S)


if __name__ == "__main__":
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    main_loop()
