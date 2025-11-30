import argparse
import asyncio
import csv
import os
import random
import time

import httpx
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple

from constants import (
    CITY_FILES,
    MODELS,
    TIMESTAMP,
    TARGET,
    OUT_DIR,
    FORECAST_HORIZONS_HOURS,
    CONTEXT_LENGTHS_DAYS,
)


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/forecast"

NUM_REQUESTS_DEFAULT = 500
POISSON_LAMBDA_DEFAULT = 4.0            # avg 4 req/s (for poisson mode)

RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
CLIENT_LOG = os.path.join(OUT_DIR, "client_trace_logs.csv")

# ============================================================
# Data loading
# ============================================================

def load_city_df(city_name: str, csv_path: str) -> pd.DataFrame:
    """
    Load city's CSV and return DataFrame with timestamp + target.
    We'll only use it to derive valid window indices.
    """
    df_raw = pd.read_csv(csv_path)
    df = df_raw.iloc[:, [0, 4]].copy()
    df.columns = [TIMESTAMP, TARGET]
    df["timestamp"] = pd.to_datetime(df[TIMESTAMP])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["id"] = city_name
    return df


def build_city_meta() -> Dict[str, Dict]:
    """
    For each city, store:
      - length (n)
      - allowed (context,horizon) combos that fit in the series
    """
    city_meta: Dict[str, Dict] = {}
    for city, path in CITY_FILES.items():
        df = load_city_df(city, path)
        n = len(df)
        combos: List[Tuple[int, int]] = []

        for d in CONTEXT_LENGTHS_DAYS:
            ctx_h = d * 24
            for hz in FORECAST_HORIZONS_HOURS:
                if ctx_h + hz < n:
                    combos.append((ctx_h, hz))

        if not combos:
            raise RuntimeError(f"No valid (context,horizon) combos for city={city}")

        city_meta[city] = {
            "df": df,
            "n": n,
            "combos": combos,
        }

    return city_meta


# ============================================================
# Trace generation
# ============================================================

def sample_request_params(
    i: int,
    city_meta: Dict[str, Dict],
) -> Dict:
    """
    Sample:
      - city
      - model_name
      - context_hours
      - horizon_hours
      - use_covariates
      - start_index
    Returns a dict suitable for ForecastRequest JSON.
    """
    city = random.choice(list(city_meta.keys()))
    meta = city_meta[city]
    n = meta["n"]
    combos = meta["combos"]

    model_name = random.choice(MODELS)

    context_hours, horizon_hours = random.choice(combos)

    # True/False roughly half
    use_covariates = random.random() < 0.5 if model_name == "amazon/chronos-2" else False

    # Valid start_index: [context_hours, n - horizon_hours)
    start_min = context_hours
    start_max = n - horizon_hours - 1
    if start_max <= start_min:
        start_index = start_min
    else:
        start_index = random.randint(start_min, start_max)

    req_id = f"req_{i:04d}"
    func_id = i

    payload = {
        "request_id": req_id,
        "func_id": func_id,
        "city": city,
        "model_name": model_name,
        "context_hours": context_hours,
        "horizon_hours": horizon_hours,
        "use_covariates": use_covariates,
        "start_index": start_index,
    }
    return payload


# ============================================================
# Client logging
# ============================================================

def init_client_logger() -> csv.DictWriter:
    file_exists = os.path.exists(CLIENT_LOG)
    f = open(CLIENT_LOG, "a", newline="", encoding="utf-8")
    fieldnames = [
        "send_ts",
        "recv_ts",
        "request_id",
        "func_id",
        "city",
        "model_name",
        "context_hours",
        "horizon_hours",
        "use_covariates",
        "start_index",
        "status_code",
        "client_latency_ms",
        "server_total_time_ms",
        "server_queue_time_ms",
        "server_load_time_ms",
        "server_inference_time_ms",
        "server_batch_size",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        f.flush()
    return writer


# ============================================================
# Async sending helpers
# ============================================================

async def send_one_request(
    client: httpx.AsyncClient,
    payload: Dict,
    writer: csv.DictWriter,
):
    url = BASE_URL + ENDPOINT

    send_ts = time.time()
    try:
        resp = await client.post(url, json=payload)
    except Exception as e:
        recv_ts = time.time()
        row = {
            "send_ts": send_ts,
            "recv_ts": recv_ts,
            "request_id": payload["request_id"],
            "func_id": payload["func_id"],
            "city": payload["city"],
            "model_name": payload["model_name"],
            "context_hours": payload["context_hours"],
            "horizon_hours": payload["horizon_hours"],
            "use_covariates": payload["use_covariates"],
            "start_index": payload["start_index"],
            "status_code": -1,
            "client_latency_ms": (recv_ts - send_ts) * 1000.0,
            "server_total_time_ms": None,
            "server_queue_time_ms": None,
            "server_load_time_ms": None,
            "server_inference_time_ms": None,
            "server_batch_size": None,
        }
        writer.writerow(row)
        return

    recv_ts = time.time()
    client_latency_ms = (recv_ts - send_ts) * 1000.0

    server_total = None
    server_queue = None
    server_load = None
    server_inf = None
    server_bs = None

    if resp.status_code == 200:
        data = resp.json()
        timings = data.get("timings_ms", {})
        server_total = timings.get("total_time_ms")
        server_queue = timings.get("queue_time_ms")
        server_load = timings.get("load_time_ms")
        server_inf = timings.get("inference_time_ms")
        server_bs = data.get("batch_size")

    row = {
        "send_ts": send_ts,
        "recv_ts": recv_ts,
        "request_id": payload["request_id"],
        "func_id": payload["func_id"],
        "city": payload["city"],
        "model_name": payload["model_name"],
        "context_hours": payload["context_hours"],
        "horizon_hours": payload["horizon_hours"],
        "use_covariates": payload["use_covariates"],
        "start_index": payload["start_index"],
        "status_code": resp.status_code,
        "client_latency_ms": client_latency_ms,
        "server_total_time_ms": server_total,
        "server_queue_time_ms": server_queue,
        "server_load_time_ms": server_load,
        "server_inference_time_ms": server_inf,
        "server_batch_size": server_bs,
    }
    writer.writerow(row)


async def run_burst_mode(
    num_requests: int,
    city_meta: Dict[str, Dict],
    writer: csv.DictWriter,
):
    """
    Mode (a): client does not wait between sending two requests.
    We schedule all tasks at once.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = []
        for i in range(num_requests):
            payload = sample_request_params(i, city_meta)
            task = asyncio.create_task(send_one_request(client, payload, writer))
            tasks.append(task)
        await asyncio.gather(*tasks)


async def run_poisson_mode(
    num_requests: int,
    city_meta: Dict[str, Dict],
    lam: float,
    writer: csv.DictWriter,
):
    """
    Mode (b): Poisson arrivals with rate lambda (requests per second).
    Inter-arrival times ~ Exp(lam).
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(num_requests):
            payload = sample_request_params(i, city_meta)
            await send_one_request(client, payload, writer)
            # sleep for next inter-arrival
            delay = np.random.exponential(1.0 / lam)
            await asyncio.sleep(delay)


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PM forecasting client for Part 5.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["burst", "poisson"],
        default="burst",
        help="burst: no wait; poisson: Poisson inter-arrival times",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=NUM_REQUESTS_DEFAULT,
        help="Number of requests to send",
    )
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=POISSON_LAMBDA_DEFAULT,
        help="Poisson rate (req/s) for poisson mode",
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    city_meta = build_city_meta()
    writer = init_client_logger()

    print(f"[CLIENT] Mode={args.mode}, num_requests={args.num_requests}")
    if args.mode == "burst":
        asyncio.run(run_burst_mode(args.num_requests, city_meta, writer))
    else:
        print(f"[CLIENT] Poisson lambda={args.lam} req/s")
        asyncio.run(run_poisson_mode(args.num_requests, city_meta, args.lam, writer))

    print(f"[CLIENT] Done. Logs written to {CLIENT_LOG}")


if __name__ == "__main__":
    main()
