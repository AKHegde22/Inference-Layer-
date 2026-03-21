import asyncio
import time
from dataclasses import dataclass
from openai import AsyncOpenAI

@dataclass
class BenchmarkResult:
    total_time:    float
    throughput:    float
    avg_latency:   float
    p50_latency:   float
    p95_latency:   float
    p99_latency:   float
    total_tokens:  int
    tokens_per_sec:float
    error_count:   int

async def benchmark_concurrent(prompts, concurrency, port=8000):
    aclient = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
    sem     = asyncio.Semaphore(concurrency)
    latencies, tokens_list = [], []
    error_count = 0

    async def single(prompt):
        nonlocal error_count
        async with sem:
            t0 = time.time()
            try:
                resp = await aclient.chat.completions.create(
                    model="mistral-7b",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=80, temperature=0.0,
                )
                latencies.append(time.time() - t0)
                tokens_list.append(resp.usage.completion_tokens)
            except Exception:
                error_count += 1
                latencies.append(time.time() - t0)
                tokens_list.append(0)

    t_start = time.time()
    await asyncio.gather(*[single(p) for p in prompts])
    total_time = time.time() - t_start

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    p = lambda pct: sorted_lat[max(0, int(n * pct) - 1)]
    total_tokens = sum(tokens_list)

    return BenchmarkResult(
        total_time=round(total_time, 2),
        throughput=round(len(prompts) / total_time, 2),
        avg_latency=round(sum(latencies) / n, 3),
        p50_latency=round(p(0.50), 3),
        p95_latency=round(p(0.95), 3),
        p99_latency=round(p(0.99), 3),
        total_tokens=total_tokens,
        tokens_per_sec=round(total_tokens / total_time, 1),
        error_count=error_count,
    )

def print_benchmark_report(result, concurrency):
    print(f"  Concurrency={concurrency} | {len([1]*40)} requests")
    print("  " + "-" * 44)
    rows = [
        ("Total Time",     f"{result.total_time}s"),
        ("Throughput",     f"{result.throughput} req/s"),
        ("Avg Latency",    f"{result.avg_latency}s"),
        ("P50 Latency",    f"{result.p50_latency}s"),
        ("P95 Latency",    f"{result.p95_latency}s"),
        ("P99 Latency",    f"{result.p99_latency}s"),
        ("Tokens/sec",     f"{result.tokens_per_sec}"),
        ("Errors",         f"{result.error_count}"),
    ]
    for k, v in rows:
        print(f"  {k:<18} {v:>12}")

prompts = [f"Summarize topic {i} in one sentence." for i in range(40)]
for c in [1, 4, 16]:
    result = asyncio.run(benchmark_concurrent(prompts, concurrency=c))
    print_benchmark_report(result, c)
    print()
