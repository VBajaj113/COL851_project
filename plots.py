# import pandas as pd
# import matplotlib.pyplot as plt
# import os


# result = "horizon"
# city = "Patna"
# x_col, loss_col = 'horizon_hours', f'{city}_rmse'
# x_label = "Horizon Hours"

# # File paths
# models = [
#     "chronos-2",
#     "chronos-bolt-base",
#     "chronos-bolt-mini",
#     "chronos-bolt-small",
#     "chronos-bolt-tiny",
#     "chronos-t5-mini",
#     "chronos-t5-small",
#     "chronos-t5-tiny",
# ]

# files = [f"./output/{m}_1_{result}_results.csv" for m in models]

# # Load all data
# data = {}
# for f in files:
#     df = pd.read_csv(f)
#     name = os.path.basename(f).replace(f"_1_{result}_results.csv", "")
#     data[name] = df

# # ========== 1. Combined Loss Curve ==========
# plt.figure()
# for name, df in data.items():
#     plt.plot(df[x_col], df[loss_col], label=name)

# plt.xlabel(x_label)
# plt.ylabel("RMSE Loss")
# plt.title(f"{city} RMSE Loss over {x_label}")
# plt.legend(fontsize=8)
# combined_curve_path = "./output/combined_loss_comparison.png"
# plt.savefig(combined_curve_path)
# plt.close()

# # ========== 2. Best Loss Scatter Plot ==========
# best_losses = []
# model_names = []

# for name, df in data.items():
#     best_losses.append(df[loss_col].min())
#     model_names.append(name)

# plt.figure()
# plt.scatter(model_names, best_losses)
# plt.xticks(rotation=45, ha="right")
# plt.ylabel("Best (Minimum) Loss")
# plt.title(f"Best {city} RMSE Loss over {x_label}")
# best_loss_scatter_path = "./output/best_loss_scatter.png"
# plt.tight_layout()
# plt.savefig(best_loss_scatter_path)
# plt.close()


# ====================


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file
# file_path = "./metrics_bak/summary_1s.csv"
# df = pd.read_csv(file_path)

# # Convert timestamp to seconds relative to start for cleaner x-axis
# time_sec = (df["ts_unix_ms"] - df["ts_unix_ms"].iloc[0]) / 1000.0

# # Plot 1: Batch Latency (p50) over time
# plt.figure()
# plt.plot(time_sec, df["p50"])
# plt.xlabel("Time (seconds)")
# plt.ylabel("Batch Latency (p50)")
# plt.title("Batch Latency (p50) Over Time")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./plots/batch_latency_p50_over_time.png")
# plt.close()

# # Plot 2: Total CPU % over time
# plt.figure()
# plt.plot(time_sec, df["cpu_total_pct"])
# plt.xlabel("Time (seconds)")
# plt.ylabel("Total CPU %")
# plt.title("Total CPU Usage Over Time")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./plots/total_cpu_over_time.png")
# plt.close()

# # Plot 3: Process RSS (MB) over time
# plt.figure()
# plt.plot(time_sec, df["proc_rss_mb"])
# plt.xlabel("Time (seconds)")
# plt.ylabel("Process RSS (MB)")
# plt.title("Process RSS Over Time")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./plots/process_rss_over_time.png")
# plt.close()

# "./plots plots generated"

# ====================

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load data
# file_path = "./metrics/latency.csv"
# df = pd.read_csv(file_path)

# # Parse time properly
# df["Time"] = pd.to_datetime(df["Time"], utc=True)
# df["t_sec"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()

# # Get unique models
# models = df["model"].unique()

# # ========= 1. Combined Latency Over Time (All Models) =========
# plt.figure()
# for m in models:
#     sub = df[df["model"] == m]
#     plt.plot(sub["t_sec"], sub["latency_s"], label=str(m))
# plt.xlabel("Time (seconds)")
# plt.ylabel("Latency (s)")
# plt.yscale("log")
# plt.title("Latency Over Time (All Models)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./plots/combined_latency_over_time.png")
# plt.close()

# # ========= 2. Combined CPU Usage Over Time (All Models) =========
# plt.figure()
# for m in models:
#     sub = df[df["model"] == m]
#     plt.plot(sub["t_sec"], sub["cpu_proc_pct"], label=str(m))
# plt.xlabel("Time (seconds)")
# plt.ylabel("CPU Process %")
# plt.yscale("log")
# plt.title("CPU Process Usage Over Time (All Models)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./plots/combined_cpu_proc_over_time.png")
# plt.close()

# # ========= 3. Combined RSS Over Time (All Models) =========
# plt.figure()
# for m in models:
#     sub = df[df["model"] == m]
#     plt.plot(sub["t_sec"], sub["proc_rss_mb"], label=str(m))
# plt.xlabel("Time (seconds)")
# plt.ylabel("Process RSS (MB)")
# plt.title("Process RSS Over Time (All Models)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./plots/combined_rss_over_time.png")
# plt.close()

# # ========= 4. Latency Distribution Per Model (Box Plot) =========
# plt.figure()
# latency_data = [df[df["model"] == m]["latency_s"].values for m in models]
# plt.boxplot(latency_data)
# plt.xticks(range(1, len(models) + 1), models, rotation=45, ha="right")
# plt.xlabel("Model")
# plt.ylabel("Latency (s)")
# plt.yscale("log")
# plt.title("Latency Distribution Per Model")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./plots/latency_distribution_models.png")
# plt.close()

# # ========= 5. Summary Statistics Per Model =========
# stats = []
# for m in models:
#     sub = df[df["model"] == m]
#     stats.append({
#         "model": m,
#         "latency_mean_s": sub["latency_s"].mean(),
#         "latency_median_s": sub["latency_s"].median(),
#         "latency_p95_s": sub["latency_s"].quantile(0.95),
#         "latency_max_s": sub["latency_s"].max(),
#         "cpu_proc_mean_pct": sub["cpu_proc_pct"].mean(),
#         "cpu_proc_max_pct": sub["cpu_proc_pct"].max(),
#         "rss_mean_mb": sub["proc_rss_mb"].mean(),
#         "rss_max_mb": sub["proc_rss_mb"].max(),
#         "avg_batch_size": sub["batch_size"].mean(),
#         "avg_stride": sub["stride"].mean()
#     })

# stats_df = pd.DataFrame(stats)
# stats_out_path = "./plots/model_summary_stats.csv"
# stats_df.to_csv(stats_out_path, index=False)


# ====================

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load data
# file_path = "./pi_metrics/pi_batch_benchmark.csv"
# df = pd.read_csv(file_path)

# # Parse time
# df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
# df["t_sec"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

# # Unique models
# models = df["model"].unique()

# # ---------- 1. Combined Latency Over Time ----------
# plt.figure()
# for m in models:
#     sub = df[df["model"] == m]
#     plt.plot(sub["t_sec"], sub["latency_s"], label=str(m))
# plt.xlabel("Time (seconds)")
# plt.ylabel("Latency (s)")
# plt.title("Raspberry Pi - Latency Over Time (Chronos-2)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# latency_time_path = "./plots/pi_combined_latency_over_time.png"
# plt.savefig(latency_time_path)
# plt.close()

# # ---------- 2. Throughput Over Time ----------
# plt.figure()
# for m in models:
#     sub = df[df["model"] == m]
#     plt.plot(sub["t_sec"], sub["throughput_samples_per_s"], label=str(m))
# plt.xlabel("Time (seconds)")
# plt.ylabel("Throughput (samples/s)")
# plt.title("Raspberry Pi - Throughput Over Time (Chronos-2)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# throughput_time_path = "./plots/pi_combined_throughput_over_time.png"
# plt.savefig(throughput_time_path)
# plt.close()

# # ---------- 3. Total CPU % Over Time ----------
# plt.figure()
# for m in models:
#     sub = df[df["model"] == m]
#     plt.plot(sub["t_sec"], sub["cpu_total_pct"], label=str(m))
# plt.xlabel("Time (seconds)")
# plt.ylabel("Total CPU %")
# plt.title("Raspberry Pi - Total CPU Usage Over Time")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# cpu_total_time_path = "./plots/pi_combined_cpu_total_over_time.png"
# plt.savefig(cpu_total_time_path)
# plt.close()

# # ---------- 4. Process RSS Over Time ----------
# plt.figure()
# for m in models:
#     sub = df[df["model"] == m]
#     plt.plot(sub["t_sec"], sub["proc_rss_mb"], label=str(m))
# plt.xlabel("Time (seconds)")
# plt.ylabel("Process RSS (MB)")
# plt.title("Raspberry Pi - Process RSS Over Time")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# rss_time_path = "./plots/pi_combined_rss_over_time.png"
# plt.savefig(rss_time_path)
# plt.close()

# # ---------- 5. Config-Level Aggregation ----------
# cfg_df = df.groupby(["model", "batch_size", "context_hours", "horizon_hours"]).agg(
#     latency_median_s=("latency_s", "median"),
#     latency_mean_s=("latency_s", "mean"),
#     throughput_mean=("throughput_samples_per_s", "mean"),
#     cpu_total_mean=("cpu_total_pct", "mean"),
#     proc_cpu_mean=("proc_cpu_pct", "mean"),
#     rss_mean_mb=("proc_rss_mb", "mean"),
# ).reset_index()

# # ---------- 6. Batch Size vs Median Latency ----------
# plt.figure()
# plt.plot(cfg_df["batch_size"], cfg_df["latency_median_s"])
# plt.xlabel("Batch Size")
# plt.ylabel("Median Latency (s)")
# plt.title("Raspberry Pi - Batch Size vs Median Latency")
# plt.grid(True)
# plt.tight_layout()
# batch_lat_path = "./plots/pi_batch_vs_median_latency.png"
# plt.savefig(batch_lat_path)
# plt.close()

# # ---------- 7. Throughput vs Median Latency ----------
# plt.figure()
# plt.plot(cfg_df["throughput_mean"], cfg_df["latency_median_s"])
# plt.xlabel("Mean Throughput (samples/s)")
# plt.ylabel("Median Latency (s)")
# plt.title("Raspberry Pi - Throughput vs Median Latency")
# plt.grid(True)
# plt.tight_layout()
# tp_lat_path = "./plots/pi_throughput_vs_median_latency.png"
# plt.savefig(tp_lat_path)
# plt.close()

# # ---------- 8. Save Config Stats ----------
# cfg_stats_path = "./plots/pi_config_level_summary_stats.csv"
# cfg_df.to_csv(cfg_stats_path, index=False)


# ===================

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load all three core configurations
# files = {
#     1: "./pi_metrics/pi_bench_gurgaon_no_covariates_1cores.csv",
#     2: "./pi_metrics/pi_bench_gurgaon_no_covariates_2cores.csv",
#     3: "./pi_metrics/pi_bench_gurgaon_no_covariates_3cores.csv",
#     4: "./pi_metrics/pi_bench_gurgaon_with_covariates_1cores.csv",
#     5: "./pi_metrics/pi_bench_gurgaon_with_covariates_2cores.csv",
#     6: "./pi_metrics/pi_bench_gurgaon_with_covariates_3cores.csv",
# }

# dfs = []
# for cores, path in files.items():
#     df = pd.read_csv(path)
#     df["t_sec"] = (df["timestamp"] - df["timestamp"].iloc[0])
#     df["covariates"] = "with_covariates" if "with_covariates" in path else "no_covariates"
#     df["num_cores"] = cores if cores <=3 else cores - 3
#     dfs.append(df)

# df_all = pd.concat(dfs, ignore_index=True)

# # ---------- 1. Latency vs Time ----------
# plt.figure()
# for cores in sorted(df_all["num_cores"].unique()):
#     for cov in df_all["covariates"].unique():
#         sub = df_all[(df_all["num_cores"] == cores) & (df_all["covariates"] == cov)]
#         plt.plot(sub["t_sec"], sub["latency_s"], label=f"{cores} Cores - {cov.replace('_', ' ').title()}")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Latency (s)")
# plt.title("Latency Over Time (1 vs 2 vs 3 Cores)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# latency_path = "./plots/pi_cores_latency_over_time.png"
# plt.savefig(latency_path)
# plt.close()

# # ---------- 2. CPU % vs Time ----------
# plt.figure()
# for cores in sorted(df_all["num_cores"].unique()):
#     for cov in df_all["covariates"].unique():
#         sub = df_all[(df_all["num_cores"] == cores) & (df_all["covariates"] == cov)]
#         plt.plot(sub["t_sec"], sub["proc_cpu_pct"], label=f"{cores} Cores - {cov.replace('_', ' ').title()}")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Process CPU %")
# plt.title("Process CPU Usage Over Time (1 vs 2 vs 3 Cores)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# cpu_path = "./plots/pi_cores_cpu_over_time.png"
# plt.savefig(cpu_path)
# plt.close()

# # ---------- 3. RSS vs Time ----------
# plt.figure()
# for cores in sorted(df_all["num_cores"].unique()):
#     for cov in df_all["covariates"].unique():
#         sub = df_all[(df_all["num_cores"] == cores) & (df_all["covariates"] == cov)]
#         plt.plot(sub["t_sec"], sub["proc_rss_mb"], label=f"{cores} Cores - {cov.replace('_', ' ').title()}")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Process RSS (MB)")
# plt.title("Process RSS Over Time (1 vs 2 vs 3 Cores)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# rss_path = "./plots/pi_cores_rss_over_time.png"
# plt.savefig(rss_path)
# plt.close()

# # ---------- 4. Temperature vs Time ----------
# plt.figure()
# for cores in sorted(df_all["num_cores"].unique()):
#     for cov in df_all["covariates"].unique():
#         sub = df_all[(df_all["num_cores"] == cores) & (df_all["covariates"] == cov)]
#         plt.plot(sub["t_sec"], sub["cpu_temp_c"], label=f"{cores} Cores - {cov.replace('_', ' ').title()}")
# plt.xlabel("Time (seconds)")
# plt.ylabel("CPU Temperature (°C)")
# plt.title("CPU Temperature Over Time (1 vs 2 vs 3 Cores)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# temp_path = "./plots/pi_cores_temp_over_time.png"
# plt.savefig(temp_path)
# plt.close()

# ====================

import pandas as pd
import matplotlib.pyplot as plt

method = "poisson"
client_path = f"./metrics/client_trace_logs_{method}.csv"
server_path = f"./metrics/server_logs_{method}.csv"

client_df = pd.read_csv(client_path)
server_df = pd.read_csv(server_path)

# Parse timestamps
client_df["send_ts"] = client_df["send_ts"]
client_df["recv_ts"] = client_df["recv_ts"]
server_df["server_receive_ts"] = server_df["server_receive_ts"]

# Use send_ts as time axis for client-side view
client_df = client_df.sort_values("send_ts")
t0 = client_df["send_ts"].iloc[0]
client_df["t_sec"] = (client_df["send_ts"] - t0)

# 1) End-to-end latency over time (client_latency_ms)
plt.figure()
plt.plot(client_df["t_sec"], client_df["client_latency_ms"])
plt.xlabel("Time (s)")
plt.ylabel("Client Latency (ms)")
plt.title("Client End-to-End Latency Over Time")
plt.grid(True)
plt.tight_layout()
e2e_time_path = f"./plots/client_e2e_latency_over_time_{method}.png"
plt.savefig(e2e_time_path)
plt.close()

# 2) Server total time over time (from client logs)
plt.figure()
plt.plot(client_df["t_sec"], client_df["server_total_time_ms"])
plt.xlabel("Time (s)")
plt.ylabel("Server Total Time (ms)")
plt.title("Server Total Time Over Time (from Client Logs)")
plt.grid(True)
plt.tight_layout()
server_total_time_path = f"./plots/server_total_time_over_time_{method}.png"
plt.savefig(server_total_time_path)
plt.close()

# 3) Breakdown stacked-style: mean composition of queue/load/inference
comp_df = client_df[["server_queue_time_ms", "server_load_time_ms", "server_inference_time_ms"]].mean()
comp_df.to_frame(name="mean_ms").to_csv(f"./plots/server_time_breakdown_mean_{method}.csv")

plt.figure()
plt.bar(["Queue", "Load", "Inference"], comp_df.values)
plt.ylabel("Mean Time (ms)")
plt.title("Mean Server Time Breakdown")
plt.tight_layout()
breakdown_bar_path = f"./plots/server_time_breakdown_mean_{method}.png"
plt.savefig(breakdown_bar_path)
plt.close()

# 4) Requests per second over time
# group by 1-second windows from t_sec
client_df["t_sec_floor"] = client_df["t_sec"].astype(int)
rps = client_df.groupby("t_sec_floor")["request_id"].count().reset_index()
plt.figure()
plt.plot(rps["t_sec_floor"], rps["request_id"])
plt.xlabel("Time (s)")
plt.ylabel("Requests per Second")
plt.title("Request Rate Over Time")
plt.grid(True)
plt.tight_layout()
rps_path = f"./plots/request_rate_over_time_{method}.png"
plt.savefig(rps_path)
plt.close()

# 5) Latency distribution per model (boxplot)
models = client_df["model_name"].unique()
latency_data = [client_df[client_df["model_name"] == m]["client_latency_ms"].values for m in models]

plt.figure()
plt.boxplot(latency_data, labels=models)
plt.xlabel("Model")
plt.ylabel("Client Latency (ms)")
plt.title("Latency Distribution per Model")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
latency_box_path = f"./plots/latency_distribution_per_model_{method}.png"
plt.savefig(latency_box_path)
plt.close()

# 6) Queue vs Load vs Inference over time (median in sliding windows)
# compute rolling median over time-sorted rows
client_df_sorted = client_df.sort_values("t_sec")
window = 50 if len(client_df_sorted) > 50 else max(5, len(client_df_sorted)//5)

rolling = client_df_sorted[["t_sec","server_queue_time_ms","server_load_time_ms","server_inference_time_ms"]].rolling(window=window, on="t_sec").median()

plt.figure()
plt.plot(rolling["t_sec"], rolling["server_queue_time_ms"], label="Queue")
plt.plot(rolling["t_sec"], rolling["server_load_time_ms"], label="Load")
plt.plot(rolling["t_sec"], rolling["server_inference_time_ms"], label="Inference")
plt.xlabel("Time (s)")
plt.ylabel("Median Time (ms, rolling)")
plt.title(f"Rolling Median Server Time Breakdown (window={window})")
plt.legend()
plt.grid(True)
plt.tight_layout()
rolling_breakdown_path = f"./plots/rolling_median_server_time_breakdown_{method}.png"
plt.savefig(rolling_breakdown_path)
plt.close()

# 7) Join client + server logs on request_id for consistency checks and correlation (optional)
merged = pd.merge(
    client_df,
    server_df,
    on=["request_id", "func_id", "city", "model_name", "context_hours", "horizon_hours", "use_covariates", "start_index"],
    how="inner",
    suffixes=("_client", "_server")
)

# Correlation between client_latency_ms and total_time_ms (server)
corr = merged["client_latency_ms"].corr(merged["total_time_ms"])
with open(f"./plots/latency_correlation_{method}.txt", "w") as f:
    f.write(f"Correlation between client_latency_ms and server total_time_ms: {corr:.4f}\n")

# RPS vs Time
merged["t_sec_floor"] = merged["send_ts"].astype(int)
rps_merged = merged.groupby("t_sec_floor")["request_id"].count().reset_index()
rps_merged.to_csv(f"./plots/merged_request_rate_over_time_{method}.csv", index=False)