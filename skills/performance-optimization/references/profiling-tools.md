# Profiling Tools Deep Dive

How to find where time actually goes, per runtime, and how to read the output. Two families
of profilers: **sampling** (py-spy, `node --cpu-prof`, `perf`) attach with low overhead and
see native frames — safe on production; **instrumentation** (`cProfile`) counts every call
exactly but slows the program down — best offline.

## Python

### cProfile (offline, instrumentation)

```bash
python -m cProfile -s cumulative my_script.py
python -m cProfile -o out.prof -m myapp.cli run
python -m pstats out.prof
```

Read the table sorted by cumulative time: large `tottime` rows are CPU hogs; large
`cumtime` spread across many calls points at a hot library call site (json, regex,
datetime). cProfile only sees Python frames — C extensions (numpy, psycopg) and syscalls
are invisible; combine with `perf` when native code is involved.

### py-spy (sampling, production-safe)

```bash
pip install py-spy
py-spy dump --pid 1234                 # one snapshot of current stacks
py-spy record --pid 1234 -o flame.svg  # 30s sample -> flame graph
py-spy record -o profile.json --format speedscope -p 1234
```

Attaches without restarting the process. Convert to a shareable flame graph with
speedscope.app or open the SVG directly.

### Memory

```bash
python -m tracemalloc my_script.py        # top allocation sites
pip install memory_profiler && mprof run my_script.py && mprof plot
```

`mprof plot` shows memory over time — a monotonically rising staircase with no plateau is
the signature of a leak; steps that reset are normal GC cycles.

## Node.js / TypeScript

### Built-in V8 profiler

```bash
node --cpu-prof --cpu-prof-dir=./profiles app.js
node --heap-prof app.js
```

Open the `.cpuprofile` in Chrome DevTools → Performance (or speedscope). `--heap-prof`
produces a heap snapshot; compare two snapshots taken at different times to find leaks.

### clinic.js (zero-config suites)

```bash
npx clinic doctor -- node app.js    # CPU, memory, event-loop delay
npx clinic flame -- node app.js     # flame graph from the same run
```

**Event-loop delay** is the Node-specific number to watch: sustained > 50-100 ms means the
loop is blocked — sync CPU work or a synchronous filesystem call in a hot path.

## Browser (Chrome DevTools)

- **Performance panel** — record an interaction or page load; read the bottom-up / flame
  chart. Look for long tasks (red bars > 50 ms), forced reflow (layout blocks after
  script), and main-thread idle gaps waiting on network.
- **Network panel** — the waterfall shows render-blocking resources (CSS in `<head>`,
  synchronous scripts), slow images, and un-cached requests.
- **Memory panel** — heap snapshots before/after an action; look for objects that should be
  garbage but survive (detached DOM nodes, listeners holding closures).
- **Lighthouse** — automated audit with a single score; good for regression checks, not for
  root-cause analysis.

## System-level

```bash
perf record -g -p <pid> && perf report     # Linux CPU sampling (native + JIT)
strace -p <pid> -c                          # syscall counts — waiting on I/O?
vmstat 1                                    # run queue, context switches, iowait
iostat -x 1                                 # disk utilization and await
```

`vmstat`'s `r` column (running threads) above the CPU count means saturation; high `wa`
(iowait) means disk-bound work.

## Reading flame graphs

- **Width** = time spent. A wide top frame is where time actually goes.
- **Flat** graphs (many narrow peaks) = a general pattern (allocation, serialization)
  rather than one function.
- **Tall** graphs = deep call chains; repeated identical towers are loops/recursion.
- The fix is validated when the wide frame *shrinks* between two flame graphs.

## Choosing a tool

| Situation | Tool |
|-----------|------|
| "Why is this request slow?" (prod backend) | py-spy / `node --cpu-prof` on one request |
| Slow script / CLI | cProfile / `node --cpu-prof` |
| CPU-bound in native code | `perf` |
| Event loop blocked | clinic doctor / DevTools Performance |
| Memory growing | heap snapshots, `tracemalloc`, mprof |
| Whole-page slow | Lighthouse + DevTools Performance/Network |
| Slow under concurrent load | load test (k6/locust) + profiler during the run |
