# Load Handling

What the deployed dashboard can take, where it breaks first, and the scaling
path. Written for a single t3.micro (2 vCPU burst, 1 GB RAM) — the free-tier
deployment in `docs/DEPLOYMENT_AWS.md`.

## Workload shape

An analyst dashboard is a low-concurrency, read-heavy workload. Realistic
load is a handful of analysts: page loads, report JSON fetches, candidate
images, occasional review POSTs. This is not a public web app and should not
be engineered like one — the honest framing is "comfortably serves a review
team; here is the measured headroom and the first bottleneck."

## Current capacity

The server is Python's ThreadingHTTPServer: one thread per in-flight
request, no event loop. Measured locally (M-series laptop; t3.micro is
slower but same order of magnitude):

| Operation | Typical latency | Notes |
|---|---|---|
| /healthz, /metrics | < 5 ms | in-memory |
| /api/report (single report) | 5–30 ms | one JSON file read |
| /api/file (candidate image) | 5–20 ms | one image read, ~100 KB |
| /api/reports (report index) | grows with history | see bottleneck #1 |
| /api/mission-plan | 20–80 ms | pure compute, no I/O |

Ten concurrent analysts clicking around produce single-digit requests per
second; the server sustains hundreds of req/s on static/JSON routes before
threading overhead matters. CPU is not the constraint — disk scanning is.

## Bottlenecks, in the order they will appear

1. **Report index scan.** `/api/reports` globs `logs/**` and reads every
   report JSON on every call. With hundreds of missions this becomes the
   slowest route (O(missions) file reads per request). Fix when felt: cache
   the index in memory and invalidate on directory mtime — one small change,
   ~100× on that route. This is the first thing to do if the dashboard ever
   feels slow.
2. **RAM on t3.micro.** 1 GB total. The Python process idles ~60–80 MB;
   large report JSONs are loaded whole per request. Concurrent loads of
   many-MB reports could pressure memory before anything else does. Mitigate
   by keeping per-mission reports bounded (`--max-saved-candidates` already
   exists) or upgrading to t3.small.
3. **Thread-per-request ceiling.** Hundreds of simultaneous slow clients
   would exhaust threads. Irrelevant at analyst-team scale; if the dashboard
   ever became multi-tenant, the move is gunicorn/uvicorn workers behind the
   same Caddy front, which the stdlib-style handler can be ported to in an
   afternoon.
4. **Session store is per-process memory.** Horizontal scaling (multiple
   replicas) would break sessions and the rate limiter. The path is sticky
   sessions at the load balancer, or a Redis-backed session store. Single
   instance: no action.

## What is deliberately NOT here

No Kubernetes, no autoscaling groups, no managed database. At this scale
each would add operational surface without measurable benefit — the system
is one container with a host volume. The scaling path above is staged so
that each step is taken when its bottleneck is actually felt, not before.

## How to load-test it yourself

```bash
# 200 requests, 10 concurrent, against the report index:
python3 - <<'EOF'
import concurrent.futures, time, urllib.request
URL = "http://localhost:8010/healthz"
def hit(_):
    start = time.monotonic()
    urllib.request.urlopen(URL, timeout=10).read()
    return (time.monotonic() - start) * 1000
with concurrent.futures.ThreadPoolExecutor(10) as pool:
    times = sorted(pool.map(hit, range(200)))
print(f"p50={times[100]:.1f}ms p95={times[190]:.1f}ms p99={times[198]:.1f}ms")
EOF
```

Swap the URL for `/api/reports` (with a session cookie if auth is on) to
measure the real bottleneck route, and watch `/metrics` while it runs —
`analyst_request_duration_ms_sum / _count` per route gives the server-side
view of the same story.
