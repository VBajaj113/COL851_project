# metrics/metrics_logger.py
import csv, os, psutil, time
from datetime import datetime, timezone
from constants import LATENCY_HEADERS


class CsvMetricsLogger:
    def __init__(self, path: str, header: list[str]):
        self.path = path
        self.header = header
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.header)
        self._proc = psutil.Process(os.getpid())

    @staticmethod
    def iso_now():
        return datetime.now(timezone.utc).isoformat()

    def log_inference(self, *, city, model, context_hours, horizon_hours, latency_s: float, batch_size = 1, stride = 1):
        cpu = self._proc.cpu_percent(interval=0.0)
        rss_mb = self._proc.memory_info().rss / (1024*1024)
        row = {
            "Time": self.iso_now(),
            "city": city,
            "model": model,
            "context_hours": int(context_hours),
            "horizon_hours": int(horizon_hours),
            "latency_s": round(latency_s, 6),
            "cpu_proc_pct": round(cpu, 2),
            "proc_rss_mb": round(rss_mb, 2),
            "batch_size": batch_size,
            "stride": stride
        }

        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.header)
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
