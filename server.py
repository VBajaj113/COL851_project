import asyncio
import csv
import gc
import json
import os
import queue
import threading
import time
import torch

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Union

from chronos import Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline
from constants import (
    CITY_FILES,
    OUT_DIR,
    MODELS,
    TIMESTAMP,
    TARGET,
)

import warnings
warnings.filterwarnings("ignore")

# --- FIX 1: Disable Torch Dynamo / Compile ---
# This prevents the "meta device" tracing error on Windows CPU
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ============================================================
# CONFIG (Part 5)
# ============================================================

DEVICE = "cpu"

# Memory limiting: how many models can be loaded at once
ENABLE_MEMORY_LIMIT = True
MAX_LOADED_MODELS = 2

# Batching (optimization on/off)
ENABLE_BATCHING = True
MAX_BATCH_SIZE = 1024
MAX_BATCH_WAIT_S = 0.1  # batch collection window

# Log file for server metrics
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE_SERVER = os.path.join(OUT_DIR, "server_logs.csv")

# ============================================================
# Logging infra
# ============================================================

class CsvLoggerThread(threading.Thread):
    def __init__(self, log_queue: queue.Queue, filename: str, fieldnames):
        super().__init__(daemon=True)
        self.log_queue = log_queue
        self.filename = filename
        self.fieldnames = fieldnames

    def run(self):
        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
        file_exists = os.path.exists(self.filename)
        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
                f.flush()

            while True:
                record = self.log_queue.get()
                if record is None:
                    break
                writer.writerow(record)
                f.flush()


server_log_queue = queue.Queue()
server_logger = CsvLoggerThread(
    server_log_queue,
    LOG_FILE_SERVER,
    fieldnames=[
        "server_receive_ts",
        "request_id",
        "func_id",
        "city",
        "model_name",
        "context_hours",
        "horizon_hours",
        "use_covariates",
        "start_index",
        "queue_time_ms",
        "load_time_ms",
        "inference_time_ms",
        "total_time_ms",
        "batch_size",
    ],
)
server_logger.start()


def log_server(record):
    server_log_queue.put(record)


def elapsed_ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


# ============================================================
# Data loading
# ============================================================

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

    df = df_raw.iloc[:, [0, 1, 2, 3, 4]].copy()
    df.columns = [TIMESTAMP, "rh", "temp", "wind", TARGET]

    df["timestamp"] = pd.to_datetime(df[TIMESTAMP])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["id"] = city_name
    df = df[["id", "timestamp", TARGET, "rh", "temp", "wind"]]
    df = df.rename(columns={TARGET: "target"})

    return df


# Global in-memory data
city_data: Dict[str, pd.DataFrame] = {}

# ============================================================
# Chronos model management (with memory limit)
# ============================================================

def load_chronos_model(model_name: str, device: str = "cpu"):
    print(f"[MODEL] Loading '{model_name}' on device '{device}'...")
    if "chronos-2" in model_name:
        pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device,
            low_cpu_mem_usage=False,
            dtype=torch.float32,
        )
    elif "chronos-bolt" in model_name:
        pipeline = ChronosBoltPipeline.from_pretrained(
            model_name,
            device_map=device,
            low_cpu_mem_usage=False,
            dtype=torch.float32,
        )
    elif "chronos-t5" in model_name:
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            low_cpu_mem_usage=False,
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unknown model type for '{model_name}'")
    
    # pipeline.model.to(device)
    # pipeline.model.eval()

    return pipeline


models: Dict[str, "ForecastModel"] = {}
global_cond = threading.Condition()
num_loaded = 0


@dataclass
class BatchJob:
    request_id: str
    func_id: int
    city: str
    model_name: str
    context_hours: int
    horizon_hours: int
    use_covariates: bool
    start_index: int
    enqueue_time: float
    future: asyncio.Future


@dataclass
class ForecastModel:
    name: str

    pipeline: Optional[Union[Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline]] = field(default=None, init=False)
    active_count: int = field(default=0, init=False)

    request_queue: Optional[asyncio.Queue] = field(default=None, init=False)
    worker_task: Optional[asyncio.Task] = field(default=None, init=False)

    def acquire_use(self):
        global global_cond
        with global_cond:
            self.active_count += 1

    def release_use(self):
        global global_cond
        with global_cond:
            self.active_count = max(0, self.active_count - 1)
            global_cond.notify_all()

    def ensure_loaded(self) -> float:
        """
        Hard memory limit:
        - At most MAX_LOADED_MODELS models with pipeline != None.
        - If limit is hit:
            * Try to evict some other idle model (active_count == 0).
            * If none idle, WAIT until someone finishes and retry.
        """
        global num_loaded, global_cond, models, ENABLE_MEMORY_LIMIT, MAX_LOADED_MODELS

        start = time.perf_counter()

        with global_cond:
            if self.pipeline is not None:
                return 0.0

            while True:
                if self.pipeline is not None:
                    return 0.0

                if not ENABLE_MEMORY_LIMIT:
                    num_loaded += 1
                    break

                if num_loaded < MAX_LOADED_MODELS:
                    num_loaded += 1
                    break

                idle_model: Optional[ForecastModel] = None
                for other in models.values():
                    if other is self:
                        continue
                    if other.pipeline is not None and other.active_count == 0:
                        idle_model = other
                        break

                if idle_model is not None:
                    print(f"[MEM] Evicting idle model '{idle_model.name}'")
                    idle_model.pipeline = None
                    num_loaded = max(0, num_loaded - 1)
                    gc.collect()
                    continue

                global_cond.wait()

        try:
            pipeline = load_chronos_model(self.name, device=DEVICE)
        except Exception:
            with global_cond:
                num_loaded = max(0, num_loaded - 1)
                global_cond.notify_all()
            raise

        with global_cond:
            self.pipeline = pipeline
            global_cond.notify_all()

        end = time.perf_counter()
        return elapsed_ms(start, end)

    def unload(self):
        global num_loaded, global_cond
        with global_cond:
            if self.pipeline is not None:
                print(f"[MEM] Unloading model '{self.name}'")
                self.pipeline = None
                num_loaded = max(0, num_loaded - 1)
                gc.collect()
                global_cond.notify_all()

    # ------------------ forecasting helpers ------------------

    def _build_context_and_future(
        self,
        city: str,
        context_hours: int,
        horizon_hours: int,
        start_index: int,
        use_covariates: bool,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Build context_df (always) and future_df (if use_covariates).
        start_index: index in city_data[city] where forecast horizon begins.
        """
        df = city_data[city]
        n = len(df)

        ctx = context_hours
        hz = horizon_hours

        # bounds check (you should ensure valid in client trace)
        if start_index < ctx or start_index + hz > n:
            raise ValueError(f"Invalid start_index={start_index} for city={city}")

        context_slice = df.iloc[start_index - ctx : start_index].copy()
        future_slice = df.iloc[start_index : start_index + hz].copy()

        # context always needs target, id, timestamp (+ covariates if enabling them as history)
        if use_covariates:
            context_df = context_slice.copy()
            future_df = future_slice.drop(columns=["target"]).copy()
        else:
            context_df = context_slice[["id", "timestamp", "target"]].copy()
            future_df = None

        return context_df, future_df

    def forecast_batch(self, jobs: List[BatchJob]) -> Dict[str, List[float]]:
        """
        Run a batch of jobs for this model and return:
        { request_id: [prediction values] }
        """
        assert self.pipeline is not None, "Model not loaded"
        pipe = self.pipeline

        # For batching, we group jobs by (use_covariates, horizon_hours)
        # to keep prediction_length consistent per sub-batch.
        groups: Dict[Tuple[bool, int], List[BatchJob]] = {}
        for j in jobs:
            key = (j.use_covariates, j.horizon_hours)
            groups.setdefault(key, []).append(j)

        results: Dict[str, List[float]] = {}

        for (use_covariates, horizon_hours), group_jobs in groups.items():
            # Build big context_df (and future_df if needed)
            all_context = []
            all_future = []  # only if use_covariates
            for job_idx, jb in enumerate(group_jobs):
                ctx_df, fut_df = self._build_context_and_future(
                    city=jb.city,
                    context_hours=jb.context_hours,
                    horizon_hours=jb.horizon_hours,
                    start_index=jb.start_index,
                    use_covariates=use_covariates,
                )
                # Give each job its own id so Chronos returns separate forecasts
                win_id = f"{jb.city}_{jb.request_id}"
                ctx_df["id"] = win_id
                if fut_df is not None:
                    fut_df["id"] = win_id

                all_context.append(ctx_df)
                if fut_df is not None:
                    all_future.append(fut_df)

            context_df = pd.concat(all_context, ignore_index=True)
            if use_covariates:
                future_df = pd.concat(all_future, ignore_index=True)
            else:
                future_df = None

            # Run predict_df once per group
            if use_covariates:
                pred_df = pipe.predict_df(
                    context_df,
                    future_df=future_df,
                    prediction_length=horizon_hours,
                    quantile_levels=[0.5],
                    id_column="id",
                    timestamp_column="timestamp",
                    target="target",
                )
            else:
                pred_df = pipe.predict_df(
                    context_df,
                    prediction_length=horizon_hours,
                    quantile_levels=[0.5],
                    id_column="id",
                    timestamp_column="timestamp",
                    target="target",
                )

            preds_by_id = pred_df.groupby("id")

            for jb in group_jobs:
                win_id = f"{jb.city}_{jb.request_id}"
                group = preds_by_id.get_group(win_id)
                vals = group["predictions"].to_numpy().tolist()
                results[jb.request_id] = vals

        return results


# ============================================================
# FastAPI schemas
# ============================================================

class ForecastRequest(BaseModel):
    request_id: str
    func_id: int
    city: str
    model_name: str
    context_hours: int
    horizon_hours: int
    use_covariates: bool = False
    start_index: int  # index in that city's time series where forecast horizon starts


class ForecastResponse(BaseModel):
    request_id: str
    func_id: int
    city: str
    model_name: str
    context_hours: int
    horizon_hours: int
    use_covariates: bool
    start_index: int
    predictions: List[float]
    timings_ms: Dict[str, float]
    batch_size: int


# ============================================================
# App setup
# ============================================================

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global city_data, models, num_loaded

    # Load time-series data once into memory
    city_data = {
        city: load_city_df_full(city, path)
        for city, path in CITY_FILES.items()
    }
    print("[DATA] Loaded city time-series:", list(city_data.keys()))

    # Initialize model wrappers (lazy load actual weights)
    models = {name: ForecastModel(name=name) for name in MODELS}
    num_loaded = 0

    if ENABLE_BATCHING:
        loop = asyncio.get_event_loop()
        for fm in models.values():
            fm.request_queue = asyncio.Queue()
            fm.worker_task = loop.create_task(batch_worker(fm))

    print("[SERVER] Startup complete.")


# ============================================================
# Batching worker (optimization)
# ============================================================

async def batch_worker(fm: ForecastModel):
    assert fm.request_queue is not None
    q = fm.request_queue

    while True:
        job: BatchJob = await q.get()
        batch: List[BatchJob] = [job]

        batch_collect_start = time.perf_counter()
        while len(batch) < MAX_BATCH_SIZE:
            remaining = MAX_BATCH_WAIT_S - (time.perf_counter() - batch_collect_start)
            if remaining <= 0:
                break
            try:
                next_job = await asyncio.wait_for(q.get(), timeout=remaining)
                batch.append(next_job)
            except asyncio.TimeoutError:
                break

        batch_size = len(batch)

        load_time_ms = 0.0
        queue_time_ms = 0.0
        inf_time_ms = 0.0
        outputs: Dict[str, List[float]] | None = None
        err: Exception | None = None

        fm.acquire_use()
        try:
            try:
                load_start = time.perf_counter()
                load_time_ms = await asyncio.to_thread(fm.ensure_loaded)
                load_end = time.perf_counter()

                inf_start = time.perf_counter()
                outputs = await asyncio.to_thread(fm.forecast_batch, batch)
                inf_end = time.perf_counter()

                # queue_time_ms is per-job; we’ll compute below
                inf_time_ms = elapsed_ms(inf_start, inf_end)
            except Exception as e:
                err = e
        finally:
            fm.release_use()

        # Resolve all futures, even if an error happened
        for b in batch:
            try:
                if err is not None:
                    if not b.future.done():
                        b.future.set_exception(err)
                else:
                    queue_time_ms = elapsed_ms(b.enqueue_time, load_start)
                    preds = outputs[b.request_id] if outputs is not None else []
                    if not b.future.done():
                        b.future.set_result(
                            (preds, load_time_ms, queue_time_ms, inf_time_ms, batch_size)
                        )
            finally:
                q.task_done()



# ============================================================
# Main endpoint
# ============================================================

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest):
    t0 = time.perf_counter()

    if req.city not in city_data:
        raise ValueError(f"Unknown city: {req.city}")
    if req.model_name not in models:
        raise ValueError(f"Unknown model: {req.model_name}")

    fm = models[req.model_name]

    if ENABLE_BATCHING:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        enqueue_time = time.perf_counter()
        job = BatchJob(
            request_id=req.request_id,
            func_id=req.func_id,
            city=req.city,
            model_name=req.model_name,
            context_hours=req.context_hours,
            horizon_hours=req.horizon_hours,
            use_covariates=req.use_covariates,
            start_index=req.start_index,
            enqueue_time=enqueue_time,
            future=fut,
        )
        await fm.request_queue.put(job)

        preds, load_time_ms, queue_time_ms, inf_time_ms, batch_size = await fut

    else:
        # No batching: simple sync path
        queue_time_ms = 0.0
        batch_size = 1

        fm.acquire_use()
        try:
            load_time_ms = await asyncio.to_thread(fm.ensure_loaded)

            def _single():
                ctx_df, fut_df = fm._build_context_and_future(
                    city=req.city,
                    context_hours=req.context_hours,
                    horizon_hours=req.horizon_hours,
                    start_index=req.start_index,
                    use_covariates=req.use_covariates,
                )
                pipe = fm.pipeline
                assert pipe is not None

                if req.use_covariates:
                    pred_df = pipe.predict_df(
                        ctx_df,
                        future_df=fut_df,
                        prediction_length=req.horizon_hours,
                        quantile_levels=[0.5],
                        id_column="id",
                        timestamp_column="timestamp",
                        target="target",
                    )
                else:
                    pred_df = pipe.predict_df(
                        ctx_df,
                        prediction_length=req.horizon_hours,
                        quantile_levels=[0.5],
                        id_column="id",
                        timestamp_column="timestamp",
                        target="target",
                    )

                vals = pred_df["predictions"].to_numpy().tolist()
                return vals

            inf_start = time.perf_counter()
            preds = await asyncio.to_thread(_single)
            inf_end = time.perf_counter()
        finally:
            fm.release_use()

        inf_time_ms = elapsed_ms(inf_start, inf_end)

    t_end = time.perf_counter()

    timings = {
        "queue_time_ms": queue_time_ms,
        "load_time_ms": load_time_ms,
        "inference_time_ms": inf_time_ms,
        "total_time_ms": elapsed_ms(t0, t_end),
    }

    log_server(
        {
            "server_receive_ts": time.time(),
            "request_id": req.request_id,
            "func_id": req.func_id,
            "city": req.city,
            "model_name": req.model_name,
            "context_hours": req.context_hours,
            "horizon_hours": req.horizon_hours,
            "use_covariates": req.use_covariates,
            "start_index": req.start_index,
            "batch_size": batch_size,
            **timings,
        }
    )

    return ForecastResponse(
        request_id=req.request_id,
        func_id=req.func_id,
        city=req.city,
        model_name=req.model_name,
        context_hours=req.context_hours,
        horizon_hours=req.horizon_hours,
        use_covariates=req.use_covariates,
        start_index=req.start_index,
        predictions=preds,
        timings_ms=timings,
        batch_size=batch_size,
    )

# Run with:
#   uvicorn server_pm:app --host 0.0.0.0 --port 8000
