# samplers/sys_sampler.py
import csv, os, time, psutil, sys
from datetime import datetime, timezone
from constants import (
    SYSTEM_SAMPLE_HEADERS as HEADERS, 
    SYSTEM_SAMPLE_CSV as CSV_PATH,
    SYSTEM_SAMPLE_INTERVAL_S as INTERVAL_S
)


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def ensure_header(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def main(target_pid=None, interval_s=INTERVAL_S):
    ensure_header(CSV_PATH, HEADERS)
    proc = psutil.Process(target_pid) if target_pid else None
    psutil.cpu_percent(interval=None)           # prime
    if proc: proc.cpu_percent(interval=None)    # prime

    while True:
        ts = iso_now()
        
        # system-wide CPU & mem
        cpu_total = psutil.cpu_percent(interval=None)
        ctimes = psutil.cpu_times_percent(interval=None)
        vm = psutil.virtual_memory()
        mem_used = (vm.total - vm.available) / (1024*1024)
        mem_avail = vm.available / (1024*1024)
        
        # process stats
        if proc and proc.is_running():
            p_cpu = proc.cpu_percent(interval=None)
            p_rss = proc.memory_info().rss / (1024*1024)
        else:
            p_cpu, p_rss = 0.0, 0.0

        row = [ts, round(cpu_total,2), round(ctimes.user,2), round(ctimes.system,2),
               round(mem_used,2), round(mem_avail,2),
               round(p_cpu,2), round(p_rss,2)]
        
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow(row)
            f.flush()

        time.sleep(interval_s)


if __name__ == "__main__":
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(pid)
