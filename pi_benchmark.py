import os
import time
import csv
import psutil
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

# ================= CONFIG =================

CITY_NAME = "Gurgaon"
CSV_PATH = "./data/gurgaon.csv"    # change to patna.csv if needed

# Use the same best hyperparameters as Part 1/3
BEST_CONTEXT_DAYS = 8
BEST_HORIZON_HOURS = 4
BEST_CONTEXT_HOURS = BEST_CONTEXT_DAYS * 24

CHRONOS2_MODEL_NAME = "amazon/chronos-2"  # or smaller variant if Pi struggles

# Benchmark config
DURATION_MINUTES = 30                    # run duration per experiment
OUT_DIR = "./pi_metrics"
os.makedirs(OUT_DIR, exist_ok=True)

# ================= CORE PINNING =================

def set_process_affinity(num_cores):
    """
    Pin current process to first `num_cores` CPU cores.
    Works on Linux (including Raspberry Pi).
    """
    try:
        all_cores = list(range(psutil.cpu_count()))
        allowed = all_cores[:num_cores]
        p = psutil.Process()
        p.cpu_affinity(allowed)
        print(f"Set CPU affinity to cores: {allowed}")
    except Exception as e:
        print(f"Warning: could not set CPU affinity: {e}")

# ================= TEMP READING =================

def read_cpu_temp_c():
    """
    Read CPU temperature in Celsius from sysfs (Raspberry Pi).
    """
    candidates = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    v = f.read().strip()
                return int(v) / 1000.0
            except Exception:
                pass
    return None

# ================= DATA PREP =================

def load_city_df(path, city_name="City"):
    df = pd.read_csv(path)

    # Assuming:
    # col 0 = timestamp
    # col 1 = RH
    # col 2 = Temp
    # col 3 = Wind
    # col 4 = PM2.5
    ts = pd.to_datetime(df.iloc[:, 0])
    pm = df.iloc[:, 4].astype(float)
    rh = df.iloc[:, 1].astype(float)
    temp = df.iloc[:, 2].astype(float)
    wind = df.iloc[:, 3].astype(float)

    city_df = pd.DataFrame({
        "id": city_name,
        "timestamp": ts,
        "pm": pm,
        "rh": rh,
        "temp": temp,
        "wind": wind,
    }).reset_index(drop=True)
    return city_df

def build_windows(city_df, context_hours, horizon_hours, use_covariates=True):
    """
    Pre-build a list of (context_df, future_df, actual) windows.
    We'll cycle over these during the benchmark to simulate a workload.
    """
    n = len(city_df)
    windows = []
    pm_series = city_df["pm"].values

    for start_idx in range(context_hours, n - horizon_hours):
        ctx_start = start_idx - context_hours
        ctx_end   = start_idx
        tgt_end   = start_idx + horizon_hours

        context_df = city_df.iloc[ctx_start:ctx_end].copy()

        if use_covariates:
            future_slice = city_df.iloc[start_idx:tgt_end].copy()
            future_df = future_slice.drop(columns=["pm"])
        else:
            future_df = None

        actual = pm_series[start_idx:tgt_end]
        windows.append((context_df, future_df, actual))

    return windows

# ================= BENCHMARK LOOP =================

def run_benchmark(city_df, chronos2, use_covariates, num_cores, duration_minutes):
    """
    Run continuous forecasts for `duration_minutes`, logging metrics.
    """
    set_process_affinity(num_cores)

    windows = build_windows(
        city_df, BEST_CONTEXT_HOURS, BEST_HORIZON_HOURS, use_covariates=use_covariates
    )
    if not windows:
        raise ValueError("No windows generated; check context/horizon settings.")

    p = psutil.Process(os.getpid())
    label_cov = "with_covariates" if use_covariates else "no_covariates"

    out_csv = os.path.join(
        OUT_DIR,
        f"pi_bench_{CITY_NAME.lower()}_{label_cov}_{num_cores}cores.csv"
    )

    print(f"Logging to {out_csv}")
    f = open(out_csv, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "city",
        "use_covariates",
        "num_cores",
        "latency_s",
        "proc_cpu_pct",
        "proc_rss_mb",
        "cpu_temp_c"
    ])

    start_time = time.time()
    end_time   = start_time + duration_minutes * 60
    idx = 0

    # warm-up
    print("Running warm-up...")
    ctx_df, future_df, _ = windows[0]
    _ = chronos2.predict_df(
        ctx_df,
        future_df=future_df,
        prediction_length=BEST_HORIZON_HOURS,
        quantile_levels=[0.5],
        id_column="id",
        timestamp_column="timestamp",
        target="pm",
    )
    print("Warm-up done. Starting main loop.")

    while time.time() < end_time:
        ctx_df, future_df, _ = windows[idx]
        idx = (idx + 1) % len(windows)

        t0 = time.perf_counter()
        pred_df = chronos2.predict_df(
            ctx_df,
            future_df=future_df,
            prediction_length=BEST_HORIZON_HOURS,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target="pm",
        )
        lat = time.perf_counter() - t0

        now = time.time()
        cpu_pct = p.cpu_percent(interval=None)   # percentage since last call
        rss_mb = p.memory_info().rss / (1024 * 1024)
        temp_c = read_cpu_temp_c()

        writer.writerow([
            now,
            CITY_NAME,
            int(use_covariates),
            num_cores,
            lat,
            cpu_pct,
            rss_mb,
            temp_c if temp_c is not None else ""
        ])

    f.close()
    print(f"Benchmark finished for {label_cov}, {num_cores} cores.")


def main():
    print(f"PID: {os.getpid()}")
    print("Loading city data...")
    city_df = load_city_df(CSV_PATH, city_name=CITY_NAME)
    print(f"Loaded {len(city_df)} rows for {CITY_NAME}")

    print(f"Loading Chronos-2 model: {CHRONOS2_MODEL_NAME}")
    chronos2 = Chronos2Pipeline.from_pretrained(
        CHRONOS2_MODEL_NAME,
        device_map="cpu",
        dtype=torch.float32,
    )
    print("Chronos-2 loaded.")

    # Choose which core-counts you want to test
    core_counts = [1, 2, 3]  # adjust depending on your Pi

    for num_cores in core_counts:
        # without covariates
        print(f"\n=== Benchmark: {num_cores} cores, NO covariates ===")
        run_benchmark(
            city_df,
            chronos2,
            use_covariates=False,
            num_cores=num_cores,
            duration_minutes=DURATION_MINUTES,
        )

        # with covariates
        print(f"\n=== Benchmark: {num_cores} cores, WITH covariates ===")
        run_benchmark(
            city_df,
            chronos2,
            use_covariates=True,
            num_cores=num_cores,
            duration_minutes=DURATION_MINUTES,
        )


if __name__ == "__main__":
    main()
