import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from sklearn.metrics import mean_squared_error
import warnings
import os
import time
from logger import CsvMetricsLogger
from constants import *
warnings.filterwarnings('ignore')

print(f"PID: {os.getpid()}")
time.sleep(4)


# Extract model name for file naming
model_short_name = MODEL_NAME.split('/')[-1]  # e.g., "chronos-t5-small"

# Load the data
print("Loading data...")
gurgaon_df = pd.read_csv(GURGAON_FILE)
patna_df = pd.read_csv(PATNA_FILE)

# Extract PM 2.5 columns
gurgaon_pm = gurgaon_df.iloc[:, PM_COLUMN_INDEX].values
patna_pm = patna_df.iloc[:, PM_COLUMN_INDEX].values

print(f"Gurgaon data points: {len(gurgaon_pm)}")
print(f"Patna data points: {len(patna_pm)}")

# Load Chronos model
print(f"\nLoading Chronos model: {MODEL_NAME}")
print("This may take a few minutes on first run...")

# Use BaseChronosPipeline which works for both T5 and Bolt models
pipeline = BaseChronosPipeline.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype=torch.float32,
)

print("Model loaded successfully!")

# === Metrics logger (per-inference) ===
os.makedirs(OUT_DIR, exist_ok=True)
metrics_path = os.path.abspath(f"{OUT_DIR}/latency.csv")
metrics = CsvMetricsLogger(metrics_path, LATENCY_HEADERS)


def forecast_and_evaluate(data, context_length, forecast_horizon, 
                         max_num_windows=MAX_NUM_WINDOWS, num_samples=NUM_SAMPLES, city_name="City"):
    """
    Perform rolling window forecasting and calculate RMSE
    
    Args:
        data: Time series data
        context_length: Number of historical points to use
        forecast_horizon: Number of points to forecast
        max_num_windows: Maximum number of test windows
        num_samples: Number of sample paths to generate (unused, kept for compatibility)
    
    Returns:
        Average RMSE across all forecasting windows
    """
    rmse_scores = []
    
    # Ensure we have enough data
    total_length = len(data)
    if total_length < context_length + forecast_horizon:
        return None
    
    # Calculate maximum possible windows
    max_possible_windows = total_length - context_length - forecast_horizon + 1
    
    # Use minimum of max_num_windows and max_possible_windows
    num_windows = min(max_num_windows, max_possible_windows)
    
    # Calculate step size for even spacing
    step_size = max(1, (total_length - context_length - forecast_horizon) // num_windows)
    
    # Rolling window evaluation
    for start_idx in range(0, total_length - context_length - forecast_horizon + 1, step_size):
        if len(rmse_scores) >= num_windows:
            break
            
        # Extract context
        context = torch.tensor(data[start_idx:start_idx + context_length])
        
        # Get actual values
        actual = data[start_idx + context_length:start_idx + context_length + forecast_horizon]
        
        # Generate forecast using predict_quantiles (works for both T5 and Bolt)
        # quantiles shape: [batch_size, prediction_length, num_quantile_levels]
        # mean shape: [batch_size, prediction_length]
        t0 = time.perf_counter()
        quantiles, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=forecast_horizon,
            quantile_levels=[0.1, 0.5, 0.9],  # Get median (0.5) and uncertainty bounds
        )
        lat = time.perf_counter() - t0
        metrics.log_inference(
            city=city_name,
            model=model_short_name,
            context_hours=context_length,
            horizon_hours=forecast_horizon,
            latency_s=lat,
        )

        # Use median forecast (quantile 0.5, which is at index 1)
        # Extract for first batch item: [prediction_length]
        forecast_values = quantiles[0, :, 1].numpy()
        
        # Calculate RMSE for this window
        rmse = np.sqrt(mean_squared_error(actual, forecast_values))
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

# Experiment 1: Different context lengths with 24-hour horizon
print("\n" + "="*60)
print("EXPERIMENT 1: Context Length Analysis (24-hour horizon)")
print("="*60)

context_lengths_hours = [d * 24 for d in CONTEXT_LENGTHS_DAYS]

results_context = {
    'context_days': CONTEXT_LENGTHS_DAYS,
    'gurgaon_rmse': [],
    'patna_rmse': []
}

for context_hours, context_days in zip(context_lengths_hours, CONTEXT_LENGTHS_DAYS):
    print(f"\nTesting context length: {context_days} days ({context_hours} hours)")
    
    # Evaluate on Gurgaon
    print("  Evaluating Gurgaon...")
    gurgaon_rmse = forecast_and_evaluate(gurgaon_pm, context_hours, HORIZON_24H, city_name=CITY_GURGAON)
    results_context['gurgaon_rmse'].append(gurgaon_rmse)
    print(f"  Gurgaon RMSE: {gurgaon_rmse:.2f}")
    
    # Evaluate on Patna
    print("  Evaluating Patna...")
    patna_rmse = forecast_and_evaluate(patna_pm, context_hours, HORIZON_24H, city_name=CITY_PATNA)
    results_context['patna_rmse'].append(patna_rmse)
    print(f"  Patna RMSE: {patna_rmse:.2f}")

    time.sleep(2)  # Brief pause between context lengths

# Plot Experiment 1
plt.figure(figsize=(12, 6))
plt.plot(CONTEXT_LENGTHS_DAYS, results_context['gurgaon_rmse'], 
         marker='o', linewidth=2, markersize=8, label='Gurgaon')
plt.plot(CONTEXT_LENGTHS_DAYS, results_context['patna_rmse'], 
         marker='s', linewidth=2, markersize=8, label='Patna')
plt.xlabel('Context Length (days)', fontsize=12)
plt.ylabel('Average RMSE (μg/m³)', fontsize=12)
plt.title(f'PM 2.5 Forecasting Performance vs Context Length\n' + 
          f'(24h horizon, {model_short_name}, max_windows={MAX_NUM_WINDOWS}, samples={NUM_SAMPLES})', 
          fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Generate filename with parameters
filename_exp1 = f'context_analysis_{model_short_name}_win{MAX_NUM_WINDOWS}_samp{NUM_SAMPLES}_.png'
plots_dir1 = "context_analysis_plots"
os.makedirs(plots_dir1,exist_ok=True)
save_path1 = os.path.join(plots_dir1,filename_exp1)
plt.savefig(save_path1, dpi=300, bbox_inches='tight')
print(f"\nSaved plot: {filename_exp1}")

# Experiment 2: Different forecast horizons with 10-day context
print("\n" + "="*60)
print("EXPERIMENT 2: Forecast Horizon Analysis (10-day context)")
print("="*60)

results_horizon = {
    'horizon_hours': FORECAST_HORIZONS_HOURS,
    'gurgaon_rmse': [],
    'patna_rmse': []
}

for horizon_hours in FORECAST_HORIZONS_HOURS:
    print(f"\nTesting forecast horizon: {horizon_hours} hours")
    
    # Evaluate on Gurgaon
    print("  Evaluating Gurgaon...")
    gurgaon_rmse = forecast_and_evaluate(gurgaon_pm, CONTEXT_10DAYS_HOURS, horizon_hours, city_name=CITY_GURGAON)
    results_horizon['gurgaon_rmse'].append(gurgaon_rmse)
    print(f"  Gurgaon RMSE: {gurgaon_rmse:.2f}")
    
    # Evaluate on Patna
    print("  Evaluating Patna...")
    patna_rmse = forecast_and_evaluate(patna_pm, CONTEXT_10DAYS_HOURS, horizon_hours, city_name=CITY_PATNA)
    results_horizon['patna_rmse'].append(patna_rmse)
    print(f"  Patna RMSE: {patna_rmse:.2f}")

    time.sleep(2)

# Plot Experiment 2
plt.figure(figsize=(12, 6))
plt.plot(FORECAST_HORIZONS_HOURS, results_horizon['gurgaon_rmse'], 
         marker='o', linewidth=2, markersize=8, label='Gurgaon')
plt.plot(FORECAST_HORIZONS_HOURS, results_horizon['patna_rmse'], 
         marker='s', linewidth=2, markersize=8, label='Patna')
plt.xlabel('Forecast Horizon (hours)', fontsize=12)
plt.ylabel('Average RMSE (μg/m³)', fontsize=12)
plt.title(f'PM 2.5 Forecasting Performance vs Horizon Length\n' + 
          f'(10d context, {model_short_name}, max_windows={MAX_NUM_WINDOWS}, samples={NUM_SAMPLES})', 
          fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Generate filename with parameters
filename_exp2 = f'horizon_analysis_{model_short_name}_win{MAX_NUM_WINDOWS}_samp{NUM_SAMPLES}_.png'
plots_dir2 = "horizon_analysis_plots"
os.makedirs(plots_dir2,exist_ok=True)
save_path2 = os.path.join(plots_dir2,filename_exp2)
plt.savefig(save_path2, dpi=300, bbox_inches='tight')
print(f"\nSaved plot: {filename_exp2}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY RESULTS")
print("="*60)
print(f"\nModel: {MODEL_NAME}")
print(f"Max Windows: {MAX_NUM_WINDOWS}")
print(f"Num Samples: {NUM_SAMPLES}")

print("\nExperiment 1 - Context Length Analysis (24h horizon):")
print(f"{'Context (days)':<15} {'Gurgaon RMSE':<15} {'Patna RMSE':<15}")
print("-" * 45)
for i, days in enumerate(CONTEXT_LENGTHS_DAYS):
    print(f"{days:<15} {results_context['gurgaon_rmse'][i]:<15.2f} {results_context['patna_rmse'][i]:<15.2f}")

print("\nExperiment 2 - Forecast Horizon Analysis (10d context):")
print(f"{'Horizon (hours)':<15} {'Gurgaon RMSE':<15} {'Patna RMSE':<15}")
print("-" * 45)
for i, hours in enumerate(FORECAST_HORIZONS_HOURS):
    print(f"{hours:<15} {results_horizon['gurgaon_rmse'][i]:<15.2f} {results_horizon['patna_rmse'][i]:<15.2f}")

# Best configurations
best_context_gurgaon = CONTEXT_LENGTHS_DAYS[np.argmin(results_context['gurgaon_rmse'])]
best_context_patna = CONTEXT_LENGTHS_DAYS[np.argmin(results_context['patna_rmse'])]
best_horizon_gurgaon = FORECAST_HORIZONS_HOURS[np.argmin(results_horizon['gurgaon_rmse'])]
best_horizon_patna = FORECAST_HORIZONS_HOURS[np.argmin(results_horizon['patna_rmse'])]

print("\nBest Configurations:")
print(f"Gurgaon: Best context = {best_context_gurgaon} days, Best horizon = {best_horizon_gurgaon} hours")
print(f"Patna: Best context = {best_context_patna} days, Best horizon = {best_horizon_patna} hours")

print("\n" + "="*60)
print("Analysis complete! Check the generated PNG files:")
print(f"  - {filename_exp1}")
print(f"  - {filename_exp2}")
print("="*60)