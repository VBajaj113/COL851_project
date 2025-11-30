import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import psutil

from constants import *
from chronos import Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline
from logger import CsvMetricsLogger
from time import perf_counter, sleep
from typing import Union

import gc
import warnings
warnings.filterwarnings('ignore')


print(f"PID: {os.getpid()}")
sleep(5)

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (8, 4)

# ======================
# CONFIG FOR BENCHMARK
# ======================

# For latency / CPU benchmarking we usually want pure CPU
USE_GPU_FOR_ACCURACY = False
DEVICE_FOR_ACCURACY = "cuda" if (USE_GPU_FOR_ACCURACY and torch.cuda.is_available()) else "cpu"

# How long to run each (model, city, context, horizon) config (in seconds)
BENCHMARK_RUNTIME_S = 120.0  # tune as you like

# === Metrics logger (per-inference latency only) ===
os.makedirs(OUT_DIR, exist_ok=True)
latency_path = os.path.abspath(f"{OUT_DIR}/latency.csv")
metrics = CsvMetricsLogger(latency_path, LATENCY_HEADERS)

# === Separate system stats CSV ===
system_stats_path = os.path.abspath(f"{OUT_DIR}/system_stats.csv")
_system_stats_file_exists = os.path.exists(system_stats_path)

system_stats_file = open(system_stats_path, "a", newline="", encoding="utf-8")
system_stats_writer = csv.DictWriter(
    system_stats_file,
    fieldnames=[
        "wall_ts",
        "city",
        "model",
        "context_hours",
        "horizon_hours",
        "latency_s",
        "batch_size",
        "stride",
        "cpu_total_pct",
        "cpu_user_pct",
        "cpu_sys_pct",
        "mem_used_mb",
        "mem_avail_mb",
        "proc_cpu_pct",
        "proc_rss_mb",
    ],
)
if not _system_stats_file_exists:
    system_stats_writer.writeheader()
    system_stats_file.flush()

DRAW_PLOTS = False


def load_model(model_name: str, device: str = 'cpu') -> Union[Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline]:
    """
    Load a Chronos model given its name.
    Supports Chronos-2, Chronos-Bolt, and Chronos models.
    """

    print(f"Loading model '{model_name}' on device '{device}'...")

    if "chronos-2" in model_name:
        pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
        )
    elif "chronos-bolt" in model_name:
        pipeline = ChronosBoltPipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
        )
    else:
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
        )
    return pipeline


def load_city_df(city_name: str, csv_path: str) -> pd.DataFrame:
    """
    Load a city's CSV and return a DataFrame with columns:
    ["id", "timestamp", "target"], sorted by timestamp.
    """
    df_raw = pd.read_csv(csv_path)

    df = df_raw.iloc[:, [0, 4]].copy()
    df.columns = [TIMESTAMP, TARGET]

    # Parse datetime and sort
    df["timestamp"] = pd.to_datetime(df[TIMESTAMP])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add Chronos-compatible columns
    df["id"] = city_name
    df = df[["id", "timestamp", TARGET]]
    df = df.rename(columns={TARGET: "target"})

    return df


def evaluate_city_config(
    city_df: pd.DataFrame,
    context_hours: int,
    horizon_hours: int,
    pipeline: Union[Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline],
    max_windows: int | None = None,
    batch_size: int = 1,
    step: int = 1,
    max_runtime_s: float | None = None,
) -> np.ndarray:
    """
    Benchmark Chronos on a single city for given context & horizon.

    For each batch, we:
      - run predict_df
      - log latency via CsvMetricsLogger (latency.csv)
      - log CPU/mem stats via system_stats.csv

    Returns: 1D numpy array of RMSE values (optional sanity check).
    """
    df = city_df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    ctx = context_hours
    hz = horizon_hours

    rmse_list: list[float] = []

    # context indices: [i-ctx, i)  -> horizon indices: [i, i+hz)
    start_i = ctx
    end_i = n - hz

    base_city_id = df["id"].iloc[0]

    # Buffers for batched prediction
    batch_context_slices: list[pd.DataFrame] = []
    batch_gt_slices: list[pd.DataFrame] = []
    batch_ids: list[str] = []

    t_global_start = perf_counter()

    def time_exceeded() -> bool:
        if max_runtime_s is None:
            return False
        return (perf_counter() - t_global_start) >= max_runtime_s

    def flush_batch():
        """Run predict_df on current batch and compute RMSEs + log latency + system stats."""
        nonlocal rmse_list, batch_context_slices, batch_gt_slices, batch_ids

        if not batch_context_slices:
            return

        batch_size_actual = len(batch_ids)
        wall_ts = perf_counter()

        # ---- Chronos inference timing ----
        t0 = perf_counter()
        batch_context_df = pd.concat(batch_context_slices, ignore_index=True)

        pred_df = pipeline.predict_df(
            batch_context_df,
            prediction_length=hz,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        t1 = perf_counter()
        latency_s = t1 - t0

        # ---- Log latency only via CsvMetricsLogger (compatible schema) ----
        metrics.log_inference(
            city=base_city_id,
            model=MODEL_NAME,
            context_hours=context_hours,
            horizon_hours=horizon_hours,
            latency_s=latency_s,
            batch_size=batch_size_actual,
            stride=step,
        )

        # ---- System metrics (psutil) -> separate CSV ----
        cpu_total = psutil.cpu_percent(interval=None)
        cpu_times = psutil.cpu_times_percent(interval=None)
        mem = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        proc_cpu = proc.cpu_percent(interval=None)
        proc_mem = proc.memory_info().rss / (1024 * 1024)  # MB

        system_stats_writer.writerow(
            {
                "wall_ts": wall_ts,
                "city": base_city_id,
                "model": MODEL_NAME,
                "context_hours": context_hours,
                "horizon_hours": horizon_hours,
                "latency_s": latency_s,
                "batch_size": batch_size_actual,
                "stride": step,
                "cpu_total_pct": cpu_total,
                "cpu_user_pct": cpu_times.user,
                "cpu_sys_pct": cpu_times.system,
                "mem_used_mb": mem.used / (1024 * 1024),
                "mem_avail_mb": mem.available / (1024 * 1024),
                "proc_cpu_pct": proc_cpu,
                "proc_rss_mb": proc_mem,
            }
        )
        system_stats_file.flush()

        # ---- RMSE computation (optional, for sanity) ----
        preds_by_id = pred_df.groupby("id")

        for win_id, gt_slice in zip(batch_ids, batch_gt_slices):
            ts_pred = (
                preds_by_id.get_group(win_id)
                .set_index("timestamp")
                .loc[gt_slice["timestamp"]]
            )
            y_pred = ts_pred["predictions"].to_numpy()
            y_true = gt_slice["target"].to_numpy()
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            rmse_list.append(rmse)

            if max_windows is not None and len(rmse_list) >= max_windows:
                break

        # Clear batch buffers
        batch_context_slices = []
        batch_gt_slices = []
        batch_ids = []


    # Iterate over windows with given stride
    for i in range(start_i, end_i, step):
        if max_windows is not None and len(rmse_list) >= max_windows:
            break
        if time_exceeded():
            break

        # Select context & ground truth slices
        context_slice = df.iloc[i - ctx : i].copy()
        gt_slice = df.iloc[i : i + hz].copy()

        # Give each window its own id so Chronos returns separate forecasts
        win_id = f"{base_city_id}_{i}"
        context_slice["id"] = win_id
        gt_slice["id"] = win_id

        batch_context_slices.append(context_slice)
        batch_gt_slices.append(gt_slice)
        batch_ids.append(win_id)

        # If batch full, flush
        if len(batch_context_slices) >= batch_size:
            flush_batch()
            if max_windows is not None and len(rmse_list) >= max_windows:
                break
            if time_exceeded():
                break

    # Flush any remaining windows (small batch) if we haven't exceeded time too badly
    if batch_context_slices and not time_exceeded():
        flush_batch()

    return np.asarray(rmse_list)


# ===========================
# Load both cities
# ===========================

city_dfs = {
    city: load_city_df(city, path)
    for city, path in CITY_FILES.items()
}

# ===========================
# Benchmark loop
# ===========================

for model in MODELS:
    MODEL_NAME = model  # used inside metrics logging
    print(f"\n\n=== BENCHMARKING model: {model} ===")
    pipeline = load_model(model, device=DEVICE_FOR_ACCURACY)
    gc.collect()
    if DEVICE_FOR_ACCURACY == "cuda":
        torch.cuda.empty_cache()

    model_name_short = model.split("/")[-1]

    # --- Config set 1: context sweep at fixed horizon (e.g. 24h) ---
    print(f"\n--- Context sweep (HORIZON = {HORIZON_24H}h), runtime per config ≈ {BENCHMARK_RUNTIME_S}s ---")
    for city, df in city_dfs.items():
        for days in CONTEXT_LENGTHS_DAYS:
            ctx_hours = days * 24
            print(f"[{model_name_short}] {city}, context={days}d ({ctx_hours}h), horizon={HORIZON_24H}h ...")
            rmses = evaluate_city_config(
                city_df=df,
                context_hours=ctx_hours,
                horizon_hours=HORIZON_24H,
                pipeline=pipeline,
                max_windows=MAX_WINDOWS_PER_CONFIG,
                batch_size=BATCH_SIZE,
                step=WINDOW_STEP,
                max_runtime_s=BENCHMARK_RUNTIME_S,
            )
            if len(rmses) > 0:
                print(f"  ran {len(rmses)} windows, avg RMSE ~ {float(np.mean(rmses)):.3f}")
            else:
                print("  ran 0 windows (config too big for series?)")
            gc.collect()
            if DEVICE_FOR_ACCURACY == "cuda":
                torch.cuda.empty_cache()

    # --- Config set 2: horizon sweep at fixed context (e.g. 10 days) ---
    fixed_ctx_hours = CONTEXT_10DAYS_HOURS
    print(f"\n--- Horizon sweep (CONTEXT = {fixed_ctx_hours}h), runtime per config ≈ {BENCHMARK_RUNTIME_S}s ---")
    for city, df in city_dfs.items():
        for horizon_hours in FORECAST_HORIZONS_HOURS:
            print(f"[{model_name_short}] {city}, context={fixed_ctx_hours}h, horizon={horizon_hours}h ...")
            rmses = evaluate_city_config(
                city_df=df,
                context_hours=fixed_ctx_hours,
                horizon_hours=horizon_hours,
                pipeline=pipeline,
                max_windows=MAX_WINDOWS_PER_CONFIG,
                batch_size=BATCH_SIZE,
                step=WINDOW_STEP,
                max_runtime_s=BENCHMARK_RUNTIME_S,
            )
            if len(rmses) > 0:
                print(f"  ran {len(rmses)} windows, avg RMSE ~ {float(np.mean(rmses)):.3f}")
            else:
                print("  ran 0 windows (config too big for series?)")
            gc.collect()
            if DEVICE_FOR_ACCURACY == "cuda":
                torch.cuda.empty_cache()

    
    # --- Config 3: Effect of batch size ---
    print(f"\n--- Batch size sweep (context={BEST_CONTEXT_HOURS}h, horizon={BEST_HORIZON_HOURS}h), runtime per config ≈ {BENCHMARK_RUNTIME_S}s ---")
    for city, df in city_dfs.items():
        for batch_size in [1, 4, 8, 16, 32]:
            print(f"[{model_name_short}] {city}, context={BEST_CONTEXT_HOURS}h, horizon={BEST_HORIZON_HOURS}h, batch_size={batch_size} ...")
            rmses = evaluate_city_config(
                city_df=df,
                context_hours=BEST_CONTEXT_HOURS,
                horizon_hours=BEST_HORIZON_HOURS,
                pipeline=pipeline,
                max_windows=MAX_WINDOWS_PER_CONFIG,
                batch_size=batch_size,
                step=WINDOW_STEP,
                max_runtime_s=BENCHMARK_RUNTIME_S,
            )
            if len(rmses) > 0:
                print(f"  ran {len(rmses)} windows, avg RMSE ~ {float(np.mean(rmses)):.3f}")
            else:
                print("  ran 0 windows (config too big for series?)")
            gc.collect()
            if DEVICE_FOR_ACCURACY == "cuda":
                torch.cuda.empty_cache()


# Close the system stats file cleanly
system_stats_file.close()

print("\nBenchmarking complete.")
print("Latency logs:", latency_path)
print("System stats logs:", system_stats_path)

