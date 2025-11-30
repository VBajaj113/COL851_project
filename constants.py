# ============================================================================
# CONFIGURATION SECTION - SET ALL PARAMETERS HERE
# ============================================================================

# Model Selection
# T5 Models: "amazon/chronos-t5-tiny", "amazon/chronos-t5-mini", 
#            "amazon/chronos-t5-small", "amazon/chronos-t5-base", "amazon/chronos-t5-large"
# Bolt Models (faster): "amazon/chronos-bolt-tiny", "amazon/chronos-bolt-mini",
#                       "amazon/chronos-bolt-small", "amazon/chronos-bolt-base"
MODELS = [
    "amazon/chronos-2",
    "amazon/chronos-bolt-tiny", 
    "amazon/chronos-bolt-mini", 
    "amazon/chronos-bolt-small", 
    "amazon/chronos-bolt-base", 
    "amazon/chronos-t5-tiny", 
    "amazon/chronos-t5-mini", 
    "amazon/chronos-t5-small", 
    "amazon/chronos-t5-base",
    # "amazon/chronos-t5-large"
]
MODEL_NAME = "amazon/chronos-bolt-tiny"

# Evaluation Parameters
MAX_WINDOWS_PER_CONFIG = None  # Maximum number of test windows per evaluation or None
MAX_NUM_WINDOWS = MAX_WINDOWS_PER_CONFIG # For forecast.py compatibility
NUM_SAMPLES = 20     # Number of sample paths (deprecated - using quantiles instead)
BATCH_SIZE = 1
WINDOW_STEP = 1

BEST_HORIZON_HOURS = 4
BEST_CONTEXT_HOURS = 8 * 24
BEST_MODEL_NAME = "amazon/chronos-2"

# Data files
CITY_GURGAON = "Gurgaon"
CITY_PATNA   = "Patna"
GURGAON_FILE = './data/gurgaon.csv'
PATNA_FILE = './data/patna.csv'

CITY_FILES = {
    CITY_GURGAON: GURGAON_FILE,
    CITY_PATNA: PATNA_FILE,
}

TARGET = "calibPM"
TIMESTAMP = "From Date"

PM_COLUMN_INDEX = 4  # Column 5 is at index 4 (0-indexed)

# Experiment 1: Context Length Analysis
CONTEXT_LENGTHS_DAYS = [
    2, 
    4, 
    8, 
    10, 
    14
]
HORIZON_24H = 24  # Fixed 24-hour horizon for Experiment 1

# Experiment 2: Forecast Horizon Analysis
FORECAST_HORIZONS_HOURS = [
    4, 
    8, 
    12, 
    24, 
    48
]
CONTEXT_10DAYS_HOURS = 10 * 24  # Fixed 10-day context for Experiment 2

OUT_DIR = "./metrics"
PLOTS_DIR = "./plots"


# how long to wait before finalizing a second-bin (to allow late samples)
BIN_LAG_SEC = 2
FLUSH_EVERY_SEC = 2
AGGREGATOR_SLEEP_S = 0.2  # how long to sleep between loops
AGGREGATOR_STATE_FILE = f"{OUT_DIR}/.agg_state.json"
SUMMARY_1S_CSV = f"{OUT_DIR}/summary_1s.csv"


SYSTEM_SAMPLE_HEADERS = [
    "Time","cpu_total_pct","cpu_user_pct","cpu_sys_pct",
    "mem_used_mb","mem_avail_mb","proc_cpu_pct","proc_rss_mb"
]
SYSTEM_SAMPLE_CSV = f"{OUT_DIR}/system_sample.csv"
SYSTEM_SAMPLE_INTERVAL_S = 1


LATENCY_HEADERS = [
    "Time","city","model","context_hours","horizon_hours",
    "latency_s","cpu_proc_pct","proc_rss_mb", "batch_size", "stride"
]
LATENCY_CSV = f"{OUT_DIR}/latency.csv"


PERF_CSV = f"{OUT_DIR}/perf_raw.csv"  # output of: perf -I 1000 -x , > perf_raw.csv

NUM_CORES = [1, 2, 3]

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================