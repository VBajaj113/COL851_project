import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from constants import *
from chronos import Chronos2Pipeline
from logger import CsvMetricsLogger
from time import perf_counter, sleep
from typing import Optional

import gc
import warnings
warnings.filterwarnings("ignore")

print(f"PID: {os.getpid()}")
sleep(3)

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (8, 4)

# =========================
# Config
# =========================

USE_GPU_FOR_ACCURACY = True
DEVICE_FOR_ACCURACY = "cuda" if (USE_GPU_FOR_ACCURACY and torch.cuda.is_available()) else "cpu"

# For Part 3, use non-overlapping windows by default
# WINDOW_STEP = BEST_HORIZON_HOURS

# How many windows to evaluate (None = all)
MAX_WINDOWS_PER_CONFIG = MAX_WINDOWS_PER_CONFIG

# CDF + time-series plots
DRAW_PLOTS = True

# Metrics loggers (separate files for with/without covariates)
os.makedirs(OUT_DIR, exist_ok=True)
metrics_path_nocov = os.path.abspath(f"{OUT_DIR}/latency_part3_nocov.csv")
metrics_path_cov = os.path.abspath(f"{OUT_DIR}/latency_part3_cov.csv")

metrics_nocov = CsvMetricsLogger(metrics_path_nocov, LATENCY_HEADERS)
metrics_cov = CsvMetricsLogger(metrics_path_cov, LATENCY_HEADERS)


# =========================
# Helpers
# =========================

def load_model(device: str = "cpu") -> Chronos2Pipeline:
    """Load Chronos-2 with given device."""
    print(f"Loading Chronos-2 '{BEST_MODEL_NAME}' on device '{device}'...")
    pipeline = Chronos2Pipeline.from_pretrained(
        BEST_MODEL_NAME,
        device_map=device,
        dtype=torch.float32,
    )
    return pipeline


def load_city_df_full(city_name: str, csv_path: str) -> pd.DataFrame:
    """
    Load city's CSV and return:
    columns = ['id', 'timestamp', 'target', 'rh', 'temp', 'wind'] (example names)
    sorted by timestamp.
    Assumes:
      col0 = timestamp
      col1 = relative humidity
      col2 = temperature
      col3 = wind speed
      col4 = PM2.5 (target)
    """
    df_raw = pd.read_csv(csv_path)

    # rename for clarity
    df = df_raw.iloc[:, [0, 1, 2, 3, 4]].copy()
    df.columns = [TIMESTAMP, "rh", "temp", "wind", TARGET]

    df["timestamp"] = pd.to_datetime(df[TIMESTAMP])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["id"] = city_name
    df = df[["id", "timestamp", TARGET, "rh", "temp", "wind"]]
    df = df.rename(columns={TARGET: "target"})

    return df


def evaluate_city_config_part3(
    city_df: pd.DataFrame,
    context_hours: int,
    horizon_hours: int,
    pipeline: Chronos2Pipeline,
    use_covariates: bool,
    max_windows: Optional[int] = None,
    step: int = 1,
    metrics_logger: Optional[CsvMetricsLogger] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Part 3 evaluation for a single city, single (ctx, horizon).

    Returns
    -------
    rmse_array : np.ndarray
        RMSE for each forecast window.
    details_df : pd.DataFrame
        One row per window:
        ['city', 'start_time', 'end_time', 'rmse']
    """
    df = city_df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    ctx = context_hours
    hz = horizon_hours

    rmse_list: list[float] = []
    rows = []

    start_i = ctx
    end_i = n - hz

    base_city_id = df["id"].iloc[0]

    for i in range(start_i, end_i, step):
        if max_windows is not None and len(rmse_list) >= max_windows:
            break

        context_slice = df.iloc[i - ctx : i].copy()
        gt_slice = df.iloc[i : i + hz].copy()

        # target-only view for ground truth
        gt_ts = gt_slice["timestamp"].to_numpy()
        y_true = gt_slice["target"].to_numpy()

        t0 = perf_counter()

        if use_covariates:
            # context has target + covariates
            context_df = context_slice.copy()

            # future covariates (perfect forecast assumption): drop target
            future_df = gt_slice.drop(columns=["target"]).copy()

            pred_df = pipeline.predict_df(
                context_df,
                future_df=future_df,
                prediction_length=hz,
                quantile_levels=[0.5],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
        else:
            # ignore covariates: keep only id, timestamp, target
            context_df = context_slice[["id", "timestamp", "target"]].copy()

            pred_df = pipeline.predict_df(
                context_df,
                prediction_length=hz,
                quantile_levels=[0.5],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

        t1 = perf_counter()

        # align predictions with ground truth timestamps
        ts_pred = (
            pred_df.set_index("timestamp")
            .loc[gt_ts]
        )
        y_pred = ts_pred["predictions"].to_numpy()

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        rmse_list.append(rmse)

        rows.append({
            "city": base_city_id,
            "start_time": gt_ts[0],
            "end_time":   gt_ts[-1],
            "rmse": rmse,
        })

        if metrics_logger is not None:
            metrics_logger.log_inference(
                city=base_city_id,
                model=BEST_MODEL_NAME + ("_cov" if use_covariates else "_nocov"),
                context_hours=context_hours,
                horizon_hours=horizon_hours,
                latency_s=t1 - t0,
                batch_size=1,
                stride=step,
            )

    rmse_arr = np.asarray(rmse_list)
    details_df = pd.DataFrame(rows)
    return rmse_arr, details_df


def plot_cdf(rmse_arr: np.ndarray, city: str, use_covariates: bool, out_dir: str):
    """Plot CDF of RMSE."""
    if rmse_arr.size == 0:
        return

    x = np.sort(rmse_arr)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure()
    plt.plot(x, y, marker=".", linestyle="-")
    plt.xlabel("RMSE (µg/m³)")
    plt.ylabel("CDF")
    plt.title(f"{city.capitalize()} – RMSE CDF ({'with' if use_covariates else 'w/o'} covariates)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    tag = "with_cov" if use_covariates else "no_cov"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{city}_rmse_cdf_{tag}.png"))
    plt.close()


def plot_worst_window_timeseries(
    city_df: pd.DataFrame,
    pipeline: Chronos2Pipeline,
    context_hours: int,
    horizon_hours: int,
    details_df: pd.DataFrame,
    use_covariates: bool,
    out_dir: str,
    top_k: int = 3,
):
    """
    For top-k worst RMSE windows, plot context+future PM:
    - x-axis: timestamp
    - y-axis: ground truth PM and forecasted PM
    """
    if details_df.empty:
        return

    df = city_df.sort_values("timestamp").reset_index(drop=True)
    base_city_id = df["id"].iloc[0]

    worst = details_df.sort_values("rmse", ascending=False).head(top_k)

    for idx, row in worst.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]

        # forecast window indices
        mask_future = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        j0 = df.index[mask_future][0]
        j1 = df.index[mask_future][-1] + 1

        # context immediately before future
        context_slice = df.iloc[j0 - context_hours : j0].copy()
        gt_slice = df.iloc[j0:j1].copy()

        # do a single forecast for plotting
        if use_covariates:
            context_df = context_slice.copy()
            future_df = gt_slice.drop(columns=["target"]).copy()
            pred_df = pipeline.predict_df(
                context_df,
                future_df=future_df,
                prediction_length=horizon_hours,
                quantile_levels=[0.5],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
        else:
            context_df = context_slice[["id", "timestamp", "target"]].copy()
            pred_df = pipeline.predict_df(
                context_df,
                prediction_length=horizon_hours,
                quantile_levels=[0.5],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

        ts_context = context_slice.set_index("timestamp")["target"].tail(context_hours)
        ts_gt = gt_slice.set_index("timestamp")["target"]
        ts_pred = pred_df.set_index("timestamp")["predictions"]

        plt.figure(figsize=(10, 4))
        ts_context.plot(label="context PM2.5", alpha=0.6)
        ts_gt.plot(label="ground truth PM2.5")
        ts_pred.plot(label="forecast PM2.5")

        plt.axvline(start_time, color="k", linestyle="--", alpha=0.5)
        plt.title(
            f"{base_city_id.capitalize()} – worst window RMSE={row['rmse']:.2f} "
            f"({'with' if use_covariates else 'w/o'} covariates)"
        )
        plt.xlabel("Timestamp")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        tag = "with_cov" if use_covariates else "no_cov"
        fname = f"{base_city_id}_worst_rmse_{tag}_{idx}.png"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()


# =========================
# Main Part 3 flow
# =========================

if __name__ == "__main__":
    # Load model and data
    pipeline = load_model(device=DEVICE_FOR_ACCURACY)
    gc.collect()
    if DEVICE_FOR_ACCURACY == "cuda":
        torch.cuda.empty_cache()

    city_dfs = {
        city: load_city_df_full(city, path)
        for city, path in CITY_FILES.items()
    }

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    for city, df in city_dfs.items():
        print(f"\n=== Part 3 – {city.upper()} (ctx={BEST_CONTEXT_HOURS}h, horizon={BEST_HORIZON_HOURS}h) ===")

        # ---------- Without covariates ----------
        print("  -> Evaluating WITHOUT covariates...")
        rmse_nocov, details_nocov = evaluate_city_config_part3(
            city_df=df,
            context_hours=BEST_CONTEXT_HOURS,
            horizon_hours=BEST_HORIZON_HOURS,
            pipeline=pipeline,
            use_covariates=False,
            max_windows=MAX_WINDOWS_PER_CONFIG,
            step=WINDOW_STEP,
            metrics_logger=metrics_nocov,
        )
        print(f"     mean RMSE (no covariates): {rmse_nocov.mean():.3f} over {len(rmse_nocov)} windows")

        # ---------- With covariates ----------
        print("  -> Evaluating WITH covariates...")
        rmse_cov, details_cov = evaluate_city_config_part3(
            city_df=df,
            context_hours=BEST_CONTEXT_HOURS,
            horizon_hours=BEST_HORIZON_HOURS,
            pipeline=pipeline,
            use_covariates=True,
            max_windows=MAX_WINDOWS_PER_CONFIG,
            step=WINDOW_STEP,
            metrics_logger=metrics_cov,
        )
        print(f"     mean RMSE (with covariates): {rmse_cov.mean():.3f} over {len(rmse_cov)} windows")

        # ---------- Save per-window RMSE ----------
        out_base = f"./output/{city}_part3_ctx{BEST_CONTEXT_HOURS}_hz{BEST_HORIZON_HOURS}"
        details_nocov.to_csv(out_base + "_nocov_windows.csv", index=False)
        details_cov.to_csv(out_base + "_withcov_windows.csv", index=False)

        # ---------- Plots ----------
        if DRAW_PLOTS:
            # CDFs
            plot_cdf(rmse_nocov, city, use_covariates=False, out_dir=PLOTS_DIR)
            plot_cdf(rmse_cov, city, use_covariates=True, out_dir=PLOTS_DIR)

            # Worst windows (high RMSE, usually sudden spikes/drops)
            plot_worst_window_timeseries(
                city_df=df,
                pipeline=pipeline,
                context_hours=BEST_CONTEXT_HOURS,
                horizon_hours=BEST_HORIZON_HOURS,
                details_df=details_nocov,
                use_covariates=False,
                out_dir=PLOTS_DIR,
                top_k=3,
            )
            plot_worst_window_timeseries(
                city_df=df,
                pipeline=pipeline,
                context_hours=BEST_CONTEXT_HOURS,
                horizon_hours=BEST_HORIZON_HOURS,
                details_df=details_cov,
                use_covariates=True,
                out_dir=PLOTS_DIR,
                top_k=3,
            )

    print("\nPart 3 evaluation finished.")
    print("Per-window RMSE CSVs and plots (CDF + worst windows) are in ./output and", PLOTS_DIR)
