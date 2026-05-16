# Scaling and Autoscaler

Two orthogonal dimensions of scaling, plus an automated controller for horizontal scaling.

## Horizontal scaling

Add or remove containers of a given process type. Traffic is load-balanced across all running containers for that type.

```bash
scalingo --app my-app scale web:5:M      # 5 web containers of size M
scalingo --app my-app scale worker:3     # 3 workers (size unchanged)
```

Benefits:

- Fault tolerance — one container failure doesn't take the service down
- Near-infinite headroom — add containers as traffic grows
- Cost control — combined with autoscaling, you pay for peak capacity only when you need it

Good for stateless web servers, job workers, and anything designed for distributed workloads.

## Vertical scaling

Change the container size (memory and CPU priority).

```bash
scalingo --app my-app scale web:2:XL     # 2 containers, upgrade to XL
```

Container tiers:

| Size | Memory  | CPU priority | PID limit |
|------|---------|--------------|-----------|
| S    | 256 MB  | low          | 128       |
| M    | 512 MB  | standard     | 256       |
| L    | 1 GB    | standard     | 512       |
| XL   | 2 GB    | high         | 1024      |
| 2XL  | 4 GB    | high         | 2048      |

XL and 2XL receive roughly double the CPU priority of M/L in contested situations. Sizes beyond 2XL exist (4XL, 8XL, etc.) — contact Scalingo to enable them.

Vertical scaling suits:

- Legacy apps not designed for distribution
- Memory-hungry runtimes (JVM, Rails, Meteor)
- Predictable workloads where the cost of coordination outweighs the benefit of distribution

The two are complementary — most production apps combine both (e.g. 3 × L web containers).

## Scaling semantics

- Scaling up/down/out/in is **zero-downtime** — new containers start and pass health checks before old ones are stopped
- Starting a process type from 0 triggers a cold start (image pull, app boot)
- Stopping is fast — running containers are sent SIGTERM, have a grace period, then SIGKILL
- Manual scale operations **disable the autoscaler** for that process type if it was active

## The Scalingo Autoscaler

Built-in horizontal autoscaler. Configure per process type, based on a target metric.

### Enable via dashboard

App → Resources → select process type → Autoscale → pick metric, target, min, max.

### Enable via CLI

```bash
scalingo --app my-app autoscalers-add --container-type web \
  --metric cpu --target 70 --min 1 --max 10

scalingo --app my-app autoscalers                  # list
scalingo --app my-app autoscalers-update <autoscaler-id> --target 60
scalingo --app my-app autoscalers-remove <autoscaler-id>
scalingo --app my-app autoscalers-enable <autoscaler-id>
scalingo --app my-app autoscalers-disable <autoscaler-id>
```

### Available metrics

| Metric | What it measures | Best for | Aggregation |
|---|---|---|---|
| `cpu` | CPU % per container | General-purpose web apps | Mean across containers |
| `ram` | RAM % per container | Memory-bound apps | Mean across containers |
| `swap` | Swap usage % | Leak detection / oversized workload | Mean |
| `rpm_per_container` | Requests per minute / container | Traffic-driven scaling | Sum / n |
| `response_time` | p95 response time | User-experience-driven scaling | p95 across requests |
| `5xx` | Server errors per minute | Reliability-driven scaling | Sum per minute |

Recommended targets for router-based metrics (RPM, response time, 5xx) are computed from your app's own last-24h median — Scalingo surfaces these in the dashboard as starting points.

**Finding the right target**: scale down to 1 container, run a load test (Locust, k6, Artillery), and find the saturation point. Set the target 20–30% below saturation so the autoscaler has time to add capacity.

### Scaling rules

The autoscaler follows conservative rules to avoid thrashing:

- **Cooldown after scale-out**: 1 minute — no second scale-out allowed in that window
- **Cooldown after scale-in**: 3 minutes — cautious about removing capacity
- **Step**: normally 1 container per decision round. The exception is `rpm_per_container`, where the autoscaler can add multiple containers per round if traffic spikes steeply (still capped at your max)
- **Min/max respected**: hard floor and ceiling

Effective for moderate ramp-up and ramp-down. Not designed for sudden 10× spikes — for predictable traffic events (product launches, news mentions), manually scale up beforehand.

### Response time caveat

A rising p95 response time doesn't always mean the containers are saturated. It may reflect:

- Slow database queries
- External API latency
- GC pressure inside the runtime

Don't set `response_time`-based autoscaling without first confirming it's container saturation, not a downstream bottleneck. The autoscaler can't fix a slow database by adding web containers.

### 5xx caveat

`5xx` triggers scaling on server error rate per minute. This only helps if errors are caused by overload (502/504 from request-queue overflow). Errors from bugs or bad config will cause the autoscaler to add capacity without fixing anything.

## Request queueing and router behavior

Understanding these numbers makes autoscaler targets saner:

- Each router keeps a local request queue per app
- Queue size: **50 requests per web container**
- Once full, new requests get **503 Service Unavailable**
- Example: 2 × web containers → 100 queued requests max

Queue-full = you're undersized. Either scale or optimize response time. RPM-based autoscaling is the right tool for predictable traffic growth; response-time-based is the right tool for unpredictable slowness.

## Scaling via Terraform

```hcl
resource "scalingo_container_type" "web" {
  app    = scalingo_app.my_app.name
  name   = "web"
  amount = 2
  size   = "L"
}

resource "scalingo_container_type" "worker" {
  app    = scalingo_app.my_app.name
  name   = "worker"
  amount = 1
  size   = "M"
}
```

`terraform apply` is the scale operation. Manual CLI scales outside of Terraform will drift — either don't mix, or use `terraform refresh` followed by config updates.

## When to scale what

Quick decision guide:

- **High RAM, low CPU** → vertical scale up (bigger container)
- **High CPU, moderate RAM** → horizontal scale out (more containers)
- **Bursty traffic, predictable patterns** → autoscaler with RPM target
- **Bursty traffic, unpredictable** → autoscaler with response-time target, plus slightly higher min
- **Steady state, no budget flexibility** → manual scaling to a fixed formation, no autoscaler
- **Background job worker accumulating backlog** → horizontal scale out (more workers); autoscale on queue length is not a built-in metric, so scale manually from worker-side signals or a custom script via the API
