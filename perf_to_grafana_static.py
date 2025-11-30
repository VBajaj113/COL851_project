import argparse, csv, math, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

# ---- mapping & helpers ----

CANON = {
    "cache-references": "cache_references",
    "cache-misses": "cache_misses",
    "LLC-load-misses": "llc_load_misses",
    "instructions": "instructions",
    "cycles": "cycles",
    # extras 
    "branches": "branches",
    "branch-misses": "branch_misses",
}

def norm_event(ev: str) -> str | None:
    """
    Normalize hybrid event names:
      cpu_core/cache-misses/ -> cache_misses
      cpu_atom/instructions/ -> instructions
    """
    if not ev:
        return None
    e = ev.strip().strip("/")
    # drop hybrid prefix if present
    parts = e.split("/", 1)
    if len(parts) == 2 and parts[0] in ("cpu_core", "cpu_atom"):
        e = parts[1]
    return CANON.get(e)

def epoch_ms_now() -> int:
    return int(time.time() * 1000)

def sec_to_epoch_ms(sec_since_start: float, epoch0_ms: int) -> int:
    return epoch0_ms + int(sec_since_start * 1000.0)

def floor_to_second(ms: int) -> int:
    return (ms // 1000) * 1000

def iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()

# ---- main transform ----

def transform_perf_csv(
    input_csv: Path,
    output_csv: Path,
    epoch0_ms: int | None = None,
    include_extras: bool = False,
) -> None:

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    # per-second aggregation store
    bins: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    inferred_epoch0 = False
    first_rel_s: float | None = None

    with input_csv.open("r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            # Rows may be malformed; require at least time, value, _, event
            if len(row) < 4:
                continue
            try:
                t_s = float(row[0])
                val = float(row[1])
                ev = row[3].strip()
            except Exception:
                continue

            if first_rel_s is None:
                first_rel_s = t_s
                if epoch0_ms is None:
                    # anchor: assume first perf row corresponds to "now"
                    # so epoch0 = now - first_rel_s
                    epoch0_ms = epoch_ms_now() - int(first_rel_s * 1000.0)
                    inferred_epoch0 = True

            metric = norm_event(ev)
            if metric is None:
                continue

            ts_ms = sec_to_epoch_ms(t_s, epoch0_ms)  # absolute epoch ms
            sec_ms = floor_to_second(ts_ms)

            # sum across atom+core (multiple rows per second per metric); keep last wins for ratios
            bins[sec_ms][metric] += val

    # write output CSV sorted by time
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "Time", "ts_unix_ms",
            "cache_references", "cache_misses", "llc_load_misses",
            "instructions", "cycles",
            "ipc", "miss_rate", "llc_mpki",
        ]
        if include_extras:
            header += ["branches", "branch_misses"]
        w.writerow(header)

        for sec_ms in sorted(bins.keys()):
            d = bins[sec_ms]
            cache_ref = d.get("cache_references", 0.0)
            cache_miss = d.get("cache_misses", 0.0)
            llc_miss = d.get("llc_load_misses", 0.0)
            inst = d.get("instructions", 0.0)
            cyc = d.get("cycles", 0.0)

            # derived metrics
            ipc = (inst / cyc) if cyc else float("nan")
            miss_rate = (cache_miss / cache_ref) if cache_ref else float("nan")
            llc_mpki = (llc_miss / (inst / 1000.0)) if inst else float("nan")

            row = [
                iso_from_ms(sec_ms), sec_ms,
                int(cache_ref), int(cache_miss), int(llc_miss),
                int(inst), int(cyc),
                round(ipc, 4) if not math.isnan(ipc) else "",
                round(miss_rate, 6) if not math.isnan(miss_rate) else "",
                round(llc_mpki, 3) if not math.isnan(llc_mpki) else "",
            ]
            if include_extras:
                row += [int(d.get("branches", 0.0)), int(d.get("branch_misses", 0.0))]
            w.writerow(row)

    hint = "(inferred)" if inferred_epoch0 else "(provided)"
    print(f"Wrote {output_csv}  | base_epoch_ms={epoch0_ms} {hint}")

# ---- CLI ----

def main():
    ap = argparse.ArgumentParser(description="Convert perf -I CSV to Grafana-friendly per-second CSV.")
    ap.add_argument("--in", dest="input_csv", required=True, help="Input perf CSV (e.g., metrics_out/perf_raw.csv)")
    ap.add_argument("--out", dest="output_csv", required=True, help="Output CSV for Grafana (e.g., metrics_out/perf_perf1s.csv)")
    ap.add_argument("--epoch0-ms", type=int, default=None,
                    help="Epoch(ms) when perf timer = 0. If omitted, inferred as now - first_time_s*1000.")
    ap.add_argument("--extras", action="store_true", help="Include branches/branch_misses columns.")
    args = ap.parse_args()

    transform_perf_csv(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        epoch0_ms=args.epoch0_ms,
        include_extras=args.extras,
    )

if __name__ == "__main__":
    main()
