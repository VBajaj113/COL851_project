import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

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
# torch.set_num_threads(8)

USE_GPU_FOR_ACCURACY = True
DEVICE_FOR_ACCURACY = "cuda" if (USE_GPU_FOR_ACCURACY and torch.cuda.is_available()) else "cpu"

# === Metrics logger (per-inference) ===
os.makedirs(OUT_DIR, exist_ok=True)
metrics_path = os.path.abspath(f"{OUT_DIR}/latency.csv")
metrics = CsvMetricsLogger(metrics_path, LATENCY_HEADERS)

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
) -> np.ndarray:
    """
    Evaluate Chronos-2 on a single city for given context & horizon.

    city_df: DataFrame with ['id', 'timestamp', 'target'], hourly frequency
    context_hours: how many past hours to use as context
    horizon_hours: how many future hours to forecast
    max_windows: optional cap on number of windows to evaluate (for speed)

    Returns: 1D numpy array of RMSE values (one per window)
    """
    df = city_df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    ctx = context_hours
    hz = horizon_hours

    rmse_list: list[float] = []

    # We use an expanding sliding window:
    # context indices: [i-ctx, i)  -> horizon indices: [i, i+hz)
    start_i = ctx
    end_i = n - hz

    base_city_id = df["id"].iloc[0]

    # Buffers for batched prediction
    batch_context_slices: list[pd.DataFrame] = []
    batch_gt_slices: list[pd.DataFrame] = []
    batch_ids: list[str] = []

    def flush_batch():
        """Run predict_df on current batch and compute RMSEs."""
        nonlocal rmse_list, batch_context_slices, batch_gt_slices, batch_ids

        if not batch_context_slices:
            return

        batch_context_df = pd.concat(batch_context_slices, ignore_index=True)

        pred_df = pipeline.predict_df(
            batch_context_df,
            prediction_length=hz,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        preds_by_id = pred_df.groupby("id")

        for win_id, gt_slice in zip(batch_ids, batch_gt_slices):
            # Align timestamps
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
            t0 = perf_counter()
            flush_batch()
            t1 = perf_counter()
            metrics.log_inference(
                city=base_city_id,
                model=MODEL_NAME,
                context_hours=context_hours,
                horizon_hours=horizon_hours,
                latency_s=t1 - t0,
                batch_size=batch_size,
                stride=step,
            )
            if max_windows is not None and len(rmse_list) >= max_windows:
                break

    # Flush any remaining windows
    flush_batch()

    return np.asarray(rmse_list)


# ===========================
# Load both cities
# ===========================

city_dfs = {
    city: load_city_df(city, path)
    for city, path in CITY_FILES.items()
}


for model in MODELS:
    print(f"\n\n=== Evaluating model: {model} ===")
    pipeline = load_model(model, device=DEVICE_FOR_ACCURACY)
    gc.collect()
    torch.cuda.empty_cache()
    context_results = {city: [] for city in city_dfs}
    model_name_short = model.split("/")[-1]

    for city, df in city_dfs.items():
        print(f"\n=== {city.upper()} – context length sweep (horizon = {HORIZON_24H}h) ===")
        for days in CONTEXT_LENGTHS_DAYS:
            ctx_hours = days * 24
            rmses = evaluate_city_config(
                city_df=df,
                context_hours=ctx_hours,
                horizon_hours=HORIZON_24H,
                pipeline=pipeline,
                max_windows=MAX_WINDOWS_PER_CONFIG,
                batch_size=BATCH_SIZE,
                step=WINDOW_STEP,
            )
            mean_rmse = float(np.mean(rmses))
            context_results[city].append(mean_rmse)
            print(f"  context = {days:2d} days ({ctx_hours:4d}h): "
                f"avg RMSE = {mean_rmse:.3f} over {len(rmses)} windows")
            gc.collect()
            torch.cuda.empty_cache()

    if DRAW_PLOTS:
        plt.figure()
        for city, rmse_list in context_results.items():
            plt.plot(CONTEXT_LENGTHS_DAYS, rmse_list, marker="o", label=city.capitalize())

        plt.xlabel("Context length (days)")
        plt.ylabel("Average RMSE (µg/m³)")
        plt.title(f"Chronos-2 – Avg RMSE vs Context Length (Horizon = {HORIZON_24H}h)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{model_name_short}_{BATCH_SIZE}_{WINDOW_STEP}_context_analysis.png"))

    horizon_results = {city: [] for city in city_dfs}
    fixed_ctx_hours = CONTEXT_10DAYS_HOURS

    for city, df in city_dfs.items():
        print(f"\n=== {city.upper()} – horizon sweep (context = {CONTEXT_10DAYS_HOURS} hours) ===")
        for horizon_hours in FORECAST_HORIZONS_HOURS:
            rmses = evaluate_city_config(
                city_df=df,
                context_hours=fixed_ctx_hours,
                horizon_hours=horizon_hours,
                pipeline=pipeline,
                max_windows=MAX_WINDOWS_PER_CONFIG,
                batch_size=BATCH_SIZE,
                step=WINDOW_STEP,
            )
            mean_rmse = float(np.mean(rmses))
            horizon_results[city].append(mean_rmse)
            print(f"  horizon = {horizon_hours:2d}h: "
                f"avg RMSE = {mean_rmse:.3f} over {len(rmses)} windows")
            gc.collect()
            torch.cuda.empty_cache()

    if DRAW_PLOTS:
        plt.figure()
        for city, rmse_list in horizon_results.items():
            plt.plot(FORECAST_HORIZONS_HOURS, rmse_list, marker="o", label=city.capitalize())

        plt.xlabel("Forecast horizon (hours)")
        plt.ylabel("Average RMSE (µg/m³)")
        plt.title(f"Chronos-2 – Avg RMSE vs Horizon (Context = {CONTEXT_10DAYS_HOURS} hours)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{model_name_short}_{WINDOW_STEP}_horizon_analysis.png"))


    for city in city_dfs.keys():
        # Best context for 24h horizon
        ctx_rmse = np.array(context_results[city])
        best_ctx_idx = int(ctx_rmse.argmin())
        best_ctx_days = CONTEXT_LENGTHS_DAYS[best_ctx_idx]

        # Best horizon for 10-day context
        hz_rmse = np.array(horizon_results[city])
        best_hz_idx = int(hz_rmse.argmin())
        best_hz_hours = FORECAST_HORIZONS_HOURS[best_hz_idx]

        print(
            f"{city.capitalize():7s} | best context (24h horizon): "
            f"{best_ctx_days} days (RMSE={ctx_rmse[best_ctx_idx]:.3f}) | "
            f"best horizon (10-day context): "
            f"{best_hz_hours}h (RMSE={hz_rmse[best_hz_idx]:.3f})"
        )


    # Save results to CSV
    results_df = pd.DataFrame({
        "context_days": CONTEXT_LENGTHS_DAYS,
        **{f"{city}_rmse": rmse_list for city, rmse_list in context_results.items()},
    })
    results_df.to_csv(f"./output/{model_name_short}_{WINDOW_STEP}_context_results.csv", index=False)
    results_df = pd.DataFrame({
        "horizon_hours": FORECAST_HORIZONS_HOURS,
        **{f"{city}_rmse": rmse_list for city, rmse_list in horizon_results.items()},
    })
    results_df.to_csv(f"./output/{model_name_short}_{WINDOW_STEP}_horizon_results.csv", index=False)