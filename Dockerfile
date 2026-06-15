# Analyst dashboard container.
# Heuristic pipeline only — the [ml] extras (torch/ultralytics) are deliberately
# excluded to keep the image small enough for free-tier hosts. Run learned-model
# benchmarks on a workstation; serve the results from this container.
FROM python:3.12-slim AS base

# opencv-python-headless avoids the libGL system dependencies of full opencv.
RUN pip install --no-cache-dir numpy>=1.24 opencv-python-headless>=4.8

WORKDIR /app

# Only what the dashboard needs at runtime: the package, static assets,
# the server, and demo mission data so a fresh deploy has something to show.
COPY pyproject.toml README.md ./
COPY autonomy/ autonomy/
COPY static/ static/
COPY analyst_server.py ./
COPY demo_data/ demo_data/
COPY benchmark_data/missions/ benchmark_data/missions/

# Non-root user; logs is the only writable surface.
RUN useradd --create-home --uid 10001 analyst \
    && mkdir -p /app/logs \
    && chown -R analyst:analyst /app/logs
USER analyst

ENV ANALYST_BIND=0.0.0.0 \
    ANALYST_DASHBOARD_PORT=8010 \
    PYTHONUNBUFFERED=1

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/healthz', timeout=2)"

CMD ["python3", "analyst_server.py"]
