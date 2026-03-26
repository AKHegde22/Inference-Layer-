Project Overview
DAY 14 — CAPSTONE: INFERENCE BENCHMARK DASHBOARD
══════════════════════════════════════════════════════

You will build a complete benchmarking system that stress-tests
BOTH llama.cpp (Week 1) and vLLM (Week 2) side by side.

WHAT YOU ARE BUILDING:
  A Python benchmarking suite + rich terminal dashboard that:

  ┌──────────────────────────────────────────────────────────┐
  │  🔬 Inference Benchmark Dashboard                        │
  │  llama.cpp (port 8080)  vs  vLLM (port 8000)            │
  ├──────────────┬───────────────────┬──────────────────────┤
  │  Metric      │  llama.cpp        │  vLLM                │
  ├──────────────┼───────────────────┼──────────────────────┤
  │  TTFT p50    │  0.41s            │  0.18s               │
  │  TTFT p95    │  0.89s            │  0.31s               │
  │  TPOT        │  0.023s/tok       │  0.009s/tok          │
  │  Throughput  │  43.5 tok/s       │  112.4 tok/s         │
  │  Concurrency │  1 req            │  32 req              │
  │  Memory est. │  4.1 GB           │  5.8 GB              │
  │  Win         │  ✓ portability    │  ✓ throughput        │
  └──────────────┴───────────────────┴──────────────────────┘

MODULES:
  benchmark/
  ├── runner.py       ← sends prompts, collects raw timings
  ├── metrics.py      ← computes TTFT, TPOT, percentiles
  ├── reporter.py     ← formats and prints dashboard
  ├── comparator.py   ← llama.cpp vs vLLM comparison logic
  └── main.py         ← CLI entry point

KEY METRICS YOU WILL MEASURE:
  TTFT  = Time To First Token
          How long until the FIRST token appears.
          → Measures perceived responsiveness.
          → Dominated by prompt processing speed.

  TPOT  = Time Per Output Token
          Average time between each generated token.
          → Measures generation speed.
          → e2e_latency / output_tokens

  Throughput = output_tokens / total_wall_time
          Tokens generated per second across ALL requests.
          → Measures system capacity under load.

  Concurrency = max simultaneous requests served efficiently
          → llama.cpp: ~1 (queue-based)
          → vLLM: 32-256 (continuous batching)

THE RUNNER: TIMING EVERY REQUEST
══════════════════════════════════════════════════════

The runner sends N prompts to a server and records precise timings.
For TTFT, we MUST use streaming — otherwise we only get total time.

class BenchmarkRunner:
    def __init__(self, base_url, model="local", api_key="none"):
        self.client   = OpenAI(base_url=base_url, api_key=api_key)
        self.model    = model
        self.base_url = base_url

    def run_single(self, prompt, max_tokens=100, temperature=0.0) -> dict:
        """Stream a single request and record timings."""
        t_start   = time.perf_counter()
        t_first   = None
        tokens    = 0
        text      = ""

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                if t_first is None:
                    t_first = time.perf_counter()  # ← TTFT moment!
                tokens += 1
                text   += delta

        t_end = time.perf_counter()
        ttft  = (t_first - t_start) if t_first else (t_end - t_start)
        total = t_end - t_start
        tpot  = (total - ttft) / tokens if tokens > 1 else total

        return {
            "prompt":       prompt,
            "response":     text,
            "tokens":       tokens,
            "ttft":         ttft,
            "tpot":         tpot,
            "total_time":   total,
            "tokens_per_s": tokens / total if total > 0 else 0,
        }

    def run_batch(self, prompts, max_tokens=100, **kwargs) -> list[dict]:
        """Run prompts sequentially (baseline, no concurrency)."""
        return [self.run_single(p, max_tokens, **kwargs) for p in prompts]

    async def run_concurrent(self, prompts, max_tokens=100,
                             concurrency=8) -> list[dict]:
        """Run prompts with asyncio concurrency (stress test)."""
        from openai import AsyncOpenAI
        import asyncio
        aclient = AsyncOpenAI(base_url=self.base_url, api_key="none")
        sem     = asyncio.Semaphore(concurrency)
        results = [None] * len(prompts)

        async def fetch(i, prompt):
            async with sem:
                # Same timing logic but async
                t0 = time.perf_counter()
                t1 = None; tokens = 0; text = ""
                stream = await aclient.chat.completions.create(
                    model=self.model,
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=max_tokens, temperature=0.0, stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        if t1 is None: t1 = time.perf_counter()
                        tokens += 1; text += delta
                t2   = time.perf_counter()
                ttft = (t1 - t0) if t1 else (t2 - t0)
                tpot = (t2 - ttft - t0) / tokens if tokens > 1 else (t2-t0)
                results[i] = {"prompt":prompt,"tokens":tokens,
                              "ttft":ttft,"tpot":tpot,
                              "total_time":t2-t0,"response":text,
                              "tokens_per_s": tokens/(t2-t0) if t2>t0 else 0}

        await asyncio.gather(*[fetch(i,p) for i,p in enumerate(prompts)])
        return results
MetricsCalculator — Statistics
COMPUTING MEANINGFUL STATISTICS
══════════════════════════════════════════════════════

Raw timings alone don't tell the full story.
You need percentiles, not just averages.

WHY PERCENTILES MATTER:
  Average TTFT = 0.5s   → sounds good!
  P99 TTFT     = 8.2s   → 1% of users wait 8 seconds!
  
  Average hides outliers. P95/P99 reveal real user experience.

class MetricsCalculator:
    @staticmethod
    def percentile(values: list, p: float) -> float:
        """Compute pth percentile (0-100) of values."""
        if not values: return 0.0
        sorted_v = sorted(values)
        idx = (p / 100) * (len(sorted_v) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_v)-1)
        frac = idx - lo
        return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo])

    @staticmethod
    def compute(results: list[dict]) -> dict:
        """Compute all stats from a list of run_single() results."""
        if not results: return {}
        ttfts  = [r["ttft"]  for r in results if "ttft"  in r]
        tpots  = [r["tpot"]  for r in results if "tpot"  in r]
        times  = [r["total_time"] for r in results]
        tokens = [r["tokens"]     for r in results]
        errors = sum(1 for r in results if r.get("tokens", 0) == 0)

        calc = MetricsCalculator
        total_tokens = sum(tokens)
        total_time   = sum(times)

        return {
            # TTFT
            "ttft_p50":   calc.percentile(ttfts, 50),
            "ttft_p95":   calc.percentile(ttfts, 95),
            "ttft_p99":   calc.percentile(ttfts, 99),
            "ttft_mean":  sum(ttfts)/len(ttfts) if ttfts else 0,
            # TPOT
            "tpot_p50":   calc.percentile(tpots, 50),
            "tpot_p95":   calc.percentile(tpots, 95),
            "tpot_mean":  sum(tpots)/len(tpots) if tpots else 0,
            # Throughput
            "throughput_tok_s": total_tokens / total_time if total_time else 0,
            "requests_per_s":   len(results) / total_time if total_time else 0,
            # Summary
            "total_requests": len(results),
            "total_tokens":   total_tokens,
            "total_time_s":   total_time,
            "error_count":    errors,
            "error_rate":     errors / len(results) if results else 0,
            "avg_tokens":     total_tokens / len(results) if results else 0,
        }
Reporter & Comparator
FORMATTING THE FINAL DASHBOARD
══════════════════════════════════════════════════════

class BenchmarkReporter:
    WIDTH = 62

    def header(self, title):
        pad = (self.WIDTH - len(title) - 2) // 2
        print("=" * self.WIDTH)
        print(" " * pad + title)
        print("=" * self.WIDTH)

    def comparison_table(self, name_a, stats_a, name_b, stats_b):
        """Print side-by-side comparison of two engines."""
        metrics = [
            ("TTFT p50 (s)",       "ttft_p50",         ".3f"),
            ("TTFT p95 (s)",       "ttft_p95",         ".3f"),
            ("TTFT p99 (s)",       "ttft_p99",         ".3f"),
            ("TPOT p50 (s/tok)",   "tpot_p50",         ".4f"),
            ("TPOT mean (s/tok)",  "tpot_mean",        ".4f"),
            ("Throughput (tok/s)", "throughput_tok_s",  ".1f"),
            ("Req/s",              "requests_per_s",    ".2f"),
            ("Total tokens",       "total_tokens",      "d"),
            ("Error rate",         "error_rate",        ".1%"),
        ]
        col_w = 22
        print(f"  {'Metric':<22} {name_a:>16} {name_b:>16}")
        print("  " + "-" * (22 + 16 + 16 + 4))

        for label, key, fmt in metrics:
            a = stats_a.get(key, 0)
            b = stats_b.get(key, 0)
            # Determine winner (lower is better for latency, higher for throughput)
            higher_is_better = key in ("throughput_tok_s", "requests_per_s", "total_tokens")
            a_wins = (a < b) if not higher_is_better else (a > b)
            b_wins = (b < a) if not higher_is_better else (b > a)
            a_str  = format(a, fmt) + (" ✓" if a_wins else "  ")
            b_str  = format(b, fmt) + (" ✓" if b_wins else "  ")
            print(f"  {label:<22} {a_str:>18} {b_str:>18}")

    def verdict(self, name_a, stats_a, name_b, stats_b):
        """Print a plain-English verdict."""
        sp = stats_b["throughput_tok_s"] / max(stats_a["throughput_tok_s"], 0.001)
        lp = stats_a["ttft_p50"] / max(stats_b["ttft_p50"], 0.001)
        print()
        print(f"  Throughput:  {name_b} is {sp:.1f}x faster than {name_a}")
        print(f"  TTFT p50:    {name_a} is {lp:.1f}x {'faster' if lp>1 else 'slower'} than {name_b}")
        print()
        print("  WHEN TO USE EACH:")
        print(f"  {name_a:<16} → edge/local, no GPU, privacy, low concurrency")
        print(f"  {name_b:<16} → cloud/API, high throughput, concurrent users")

Putting It All Together
MAIN.PY — THE FULL BENCHMARK ENTRY POINT
══════════════════════════════════════════════════════

BENCHMARK PROMPTS (diverse, realistic):

  PROMPTS = [
      # Short factual (fast TTFT test)
      "What is the capital of Japan?",
      "Name three primary colors.",
      "What year was Python created?",
      # Medium reasoning (TPOT test)
      "Explain what a neural network is in 2 sentences.",
      "What are the pros and cons of microservices?",
      "Describe how TCP/IP works simply.",
      # Code generation (token volume test)
      "Write a Python function to check if a number is prime.",
      "Show a SQL query joining two tables on user_id.",
      "Write a bash one-liner to count lines in all .py files.",
      # Long output (throughput stress)
      "List 10 common HTTP status codes with descriptions.",
      "Explain 5 key differences between Python 2 and Python 3.",
      "Describe the steps to deploy a web app to production.",
  ] * 4  # repeat 4x = 48 prompts total

MAIN FUNCTION:

  import argparse, time, json
  from benchmark.runner    import BenchmarkRunner
  from benchmark.metrics   import MetricsCalculator
  from benchmark.reporter  import BenchmarkReporter

  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--llamacpp-url", default="http://localhost:8080/v1")
      parser.add_argument("--vllm-url",     default="http://localhost:8000/v1")
      parser.add_argument("--prompts",      type=int, default=48)
      parser.add_argument("--max-tokens",   type=int, default=100)
      parser.add_argument("--concurrency",  type=int, default=8)
      parser.add_argument("--output-json",  default=None)
      args = parser.parse_args()

      reporter = BenchmarkReporter()
      reporter.header("Inference Benchmark: llama.cpp vs vLLM")

      # Sequential benchmark
      print("\n[1/2] Benchmarking llama.cpp (sequential)...")
      lc_runner  = BenchmarkRunner(args.llamacpp_url)
      lc_results = lc_runner.run_batch(PROMPTS[:args.prompts], args.max_tokens)
      lc_stats   = MetricsCalculator.compute(lc_results)

      print("[2/2] Benchmarking vLLM (concurrent)...")
      import asyncio
      vl_runner  = BenchmarkRunner(args.vllm_url)
      vl_results = asyncio.run(
          vl_runner.run_concurrent(PROMPTS[:args.prompts],
                                   args.max_tokens, args.concurrency)
      )
      vl_stats = MetricsCalculator.compute(vl_results)

      # Print dashboard
      reporter.comparison_table("llama.cpp", lc_stats, "vLLM", vl_stats)
      reporter.verdict("llama.cpp", lc_stats, "vLLM", vl_stats)

      if args.output_json:
          with open(args.output_json, "w") as f:
              json.dump({"llamacpp": lc_stats, "vllm": vl_stats}, f, indent=2)
          print(f"  Results saved to {args.output_json}")

  if __name__ == "__main__":
      main()

USAGE:
  # Basic run
  python -m benchmark.main

  # Custom endpoints + save results
  python -m benchmark.main \
    --llamacpp-url http://localhost:8080/v1 \
    --vllm-url     http://localhost:8000/v1 \
    --prompts 50   --max-tokens 150 \
    --output-json  results.json

  # Higher concurrency stress test
  python -m benchmark.main --concurrency 16 --prompts 100