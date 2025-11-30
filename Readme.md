### How to run

First, set up a Python virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, in separate terminal windows, run the following commands:

1. Terminal A — forecast: `python3 forecast.py`
2. Terminal B — sys sampler: `python3 system_sampler.py $(pgrep -n -f forecast.py)` or `python3 system_sampler.py`
3. Terminal C — perf stats (Optional):
```bash
./perf_stats.sh &
python perf_to_grafana_stream.py
```
4. Terminal D — aggregate periodically (for Grafana): `python3 aggregator.py`
5. Terminal E — To run Grafana: `docker-compose up -d`
6. Terminal F — To serve the csv files over http: `cd ./metrics && python3 -m http.server 8088`
