import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional
from constants import OUT_DIR, PERF_CSV, BIN_LAG_SEC, FLUSH_EVERY_SEC

CANON = {
    "cache-references": "cache_references",
    "cache-misses": "cache_misses",
    "LLC-load-misses": "llc_load_misses",
    "instructions": "instructions",
    "cycles": "cycles",
    "branches": "branches",
    "branch-misses": "branch_misses",
}

def norm_event(ev: str) -> Optional[str]:
    """Normalize hybrid event names: 'cpu_core/cache-misses/' -> 'cache_misses'."""
    if not ev:
        return None
    e = ev.strip().strip("/")
    parts = e.split("/", 1)
    if len(parts) == 2 and parts[0] in ("cpu_core", "cpu_atom"):
        e = parts[1]
    return CANON.get(e)

def now_ms() -> int:
    return int(time.time() * 1000)

def sec_to_epoch_ms(sec_since_start: float, epoch0_ms: int) -> int:
    return epoch0_ms + int(sec_since_start * 1000.0)

def floor_to_sec(ms: int) -> int:
    return (ms // 1000) * 1000

def iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()

class TailState:
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.offset: int = 0
        self.epoch0_ms: Optional[int] = None

    def load(self):
        if self.state_file.exists():
            try:
                d = json.loads(self.state_file.read_text())
                self.offset = int(d.get("offset", 0))
                self.epoch0_ms = d.get("epoch0_ms", None)
            except Exception:
                pass

    def save(self):
        self.state_file.write_text(json.dumps({"offset": self.offset, "epoch0_ms": self.epoch0_ms}))

class Bin:
    """Aggregate per-second counters."""
    __slots__ = ("vals",)

    def __init__(self):
        # store sums per canonical metric
        self.vals: Dict[str, float] = defaultdict(float)

    def add(self, metric: str, val: float):
        self.vals[metric] += val

    def finalize(self) -> Tuple[float, float, float]:
        """Return derived: ipc, miss_rate, llc_mpki (NaN if not computable)."""
        v = self.vals
        inst = v.get("instructions", 0.0)
        cyc = v.get("cycles", 0.0)
        cref = v.get("cache_references", 0.0)
        cmiss = v.get("cache_misses", 0.0)
        llc = v.get("llc_load_misses", 0.0)

        ipc = (inst / cyc) if cyc else float("nan")
        miss_rate = (cmiss / cref) if cref else float("nan")
        llc_mpki = (llc / (inst / 1000.0)) if inst else float("nan")
        return ipc, miss_rate, llc_mpki

class PerfStreamer:
    def __init__(self, in_path: Path, out_path: Path, state_path: Path,
                 bin_lag_sec: int = 2, flush_every_sec: int = 2):
        self.in_path = in_path
        self.out_path = out_path
        self.state = TailState(state_path)
        self.state.load()
        self.bin_lag_ms = bin_lag_sec * 1000
        self.flush_every_ms = flush_every_sec * 1000

        # active bins: sec_ms -> Bin
        self.bins: Dict[int, Bin] = defaultdict(Bin)
        self.last_flush_ms = 0

        # ensure header
        if not self.out_path.exists() or self.out_path.stat().st_size == 0:
            self._write_header()

    def _write_header(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "ts_unix_ms",
                "cache_references", "cache_misses", "llc_load_misses",
                "instructions", "cycles",
                "ipc", "miss_rate", "llc_mpki"
            ])

    def tail_new_rows(self):
        """Yield new rows from perf CSV since last offset, update offset."""
        if not self.in_path.exists():
            time.sleep(0.2)
            return []

        size = self.in_path.stat().st_size
        if self.state.offset > size:
            # rotated/truncated
            self.state.offset = 0

        out = []
        with self.in_path.open("r", newline="") as f:
            f.seek(self.state.offset)
            rdr = csv.reader(f)
            # no header in perf output → good; if present, we don't rely on it
            for row in rdr:
                out.append(row)
            self.state.offset = f.tell()
        return out

    def process_rows(self, rows):
        # Determine epoch0_ms lazily at the first row
        now_epoch = now_ms()
        for r in rows:
            if len(r) < 4:
                continue
            try:
                t_s = float(r[0])  # seconds since perf start
                val = float(r[1])
                ev = r[3].strip()
            except Exception:
                continue

            if self.state.epoch0_ms is None:
                # anchor: first perf row corresponds to "now"
                self.state.epoch0_ms = now_epoch - int(t_s * 1000.0)

            metric = norm_event(ev)
            if metric is None:
                continue

            ts_ms = sec_to_epoch_ms(t_s, self.state.epoch0_ms)
            sec_ms = floor_to_sec(ts_ms)

            b = self.bins[sec_ms]
            b.add(metric, val)

    def flush_ready(self):
        """Flush bins whose second is older than now - lag."""
        if not self.bins:
            return 0

        now_epoch = now_ms()
        cutoff = now_epoch - self.bin_lag_ms
        rows = []
        for sec_ms, b in list(self.bins.items()):
            if sec_ms <= cutoff:
                v = b.vals
                ipc, miss_rate, llc_mpki = b.finalize()
                rows.append([
                    iso_from_ms(sec_ms), sec_ms,
                    int(v.get("cache_references", 0.0)),
                    int(v.get("cache_misses", 0.0)),
                    int(v.get("llc_load_misses", 0.0)),
                    int(v.get("instructions", 0.0)),
                    int(v.get("cycles", 0.0)),
                    (round(ipc, 6) if not math.isnan(ipc) else ""),
                    (round(miss_rate, 6) if not math.isnan(miss_rate) else ""),
                    (round(llc_mpki, 6) if not math.isnan(llc_mpki) else "")
                ])
                del self.bins[sec_ms]

        if rows:
            with self.out_path.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerows(rows)
        return len(rows)

    def run(self):
        print(f"[perf_stream_live] watching {self.in_path} -> writing {self.out_path}")
        try:
            while True:
                rows = self.tail_new_rows()
                if rows:
                    self.process_rows(rows)

                now_epoch = now_ms()
                if now_epoch - self.last_flush_ms >= self.flush_every_ms:
                    flushed = self.flush_ready()
                    self.state.save()
                    self.last_flush_ms = now_epoch
                    # uncomment for debug:
                    # if flushed: print(f"[perf_stream_live] flushed {flushed} rows")

                time.sleep(0.2)
        except KeyboardInterrupt:
            # final flush on exit
            self.flush_ready()
            self.state.save()
            print("\n[perf_stream_live] stopped.")

def main():
    in_path = Path(PERF_CSV) 
    out_path = Path(f"{OUT_DIR}/perf_processed.csv")
    state_file = Path( f"{OUT_DIR}/perf.state.json")

    streamer = PerfStreamer(
        in_path=in_path,
        out_path=out_path,
        state_path=state_file,
        bin_lag_sec=BIN_LAG_SEC,
        flush_every_sec=FLUSH_EVERY_SEC,
    )
    streamer.run()

if __name__ == "__main__":
    main()
