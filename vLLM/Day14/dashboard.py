class MetricsCalculator:
    @staticmethod
    def percentile(values, p):
        if not values:
            return 0.0
        sv  = sorted(values)
        idx = (p / 100.0) * (len(sv) - 1)
        lo  = int(idx)
        hi  = min(lo + 1, len(sv) - 1)
        return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])

    @staticmethod
    def compute(results):
        if not results:
            return {}
        calc   = MetricsCalculator
        ttfts  = [r["ttft"]       for r in results if r.get("tokens", 0) > 0]
        tpots  = [r["tpot"]       for r in results if r.get("tokens", 0) > 0]
        times  = [r["total_time"] for r in results]
        tokens = [r["tokens"]     for r in results]
        errors = sum(1 for r in results if r.get("tokens", 0) == 0)
        total_tokens = sum(tokens)
        total_time   = sum(times)
        return {
            "ttft_p50":          calc.percentile(ttfts, 50),
            "ttft_p95":          calc.percentile(ttfts, 95),
            "ttft_p99":          calc.percentile(ttfts, 99),
            "ttft_mean":         sum(ttfts)/len(ttfts) if ttfts else 0,
            "tpot_p50":          calc.percentile(tpots, 50),
            "tpot_p95":          calc.percentile(tpots, 95),
            "tpot_mean":         sum(tpots)/len(tpots) if tpots else 0,
            "throughput_tok_s":  total_tokens / total_time if total_time else 0,
            "requests_per_s":    len(results) / total_time if total_time else 0,
            "total_requests":    len(results),
            "total_tokens":      total_tokens,
            "total_time_s":      total_time,
            "error_count":       errors,
            "error_rate":        errors / len(results),
            "avg_tokens":        total_tokens / len(results),
        }


import random
random.seed(42)
mock_results = []
for i in range(20):
    tokens = random.randint(30, 120)
    total  = random.uniform(0.5, 3.0)
    ttft   = random.uniform(0.05, 0.5)
    tpot   = (total - ttft) / max(tokens - 1, 1)
    mock_results.append({"ttft":ttft,"tpot":tpot,"total_time":total,"tokens":tokens,"tokens_per_s":tokens/total})
mock_results.append({"ttft":0,"tpot":0,"total_time":0.1,"tokens":0})

calc  = MetricsCalculator
p_test = calc.percentile([1,2,3,4,5], 90)
print(f"percentile([1-5], 90) = {p_test:.1f}  (expected 4.6)")

stats = calc.compute(mock_results)
print(f"TTFT p50:      {stats['ttft_p50']:.3f}s")
print(f"TTFT p95:      {stats['ttft_p95']:.3f}s")
print(f"Throughput:    {stats['throughput_tok_s']:.1f} tok/s")
print(f"Error rate:    {stats['error_rate']:.1%}")
print(f"Total tokens:  {stats['total_tokens']}")

class BenchmarkReporter:
    WIDTH = 64

    def header(self, title):
        pad = (self.WIDTH - len(title) - 2) // 2
        print("=" * self.WIDTH)
        print(" " * pad + title)
        print("=" * self.WIDTH)

    def row(self, label, val_a, val_b, higher_is_better=False, fmt=".3f"):
        if fmt == ".1%":
            sa, sb = format(val_a, ".1%"), format(val_b, ".1%")
        else:
            sa, sb = format(val_a, fmt), format(val_b, fmt)
        if val_a == val_b:
            pass
        elif (not higher_is_better and val_a < val_b) or (higher_is_better and val_a > val_b):
            sa += " v"
        else:
            sb += " v"
        print(f"  {label:<24} {sa:>18} {sb:>18}")

    def comparison_table(self, name_a, stats_a, name_b, stats_b):
        print(f"  {'Metric':<24} {name_a:>18} {name_b:>18}")
        print("  " + "-" * 62)
        self.row("TTFT p50 (s)",      stats_a["ttft_p50"],         stats_b["ttft_p50"])
        self.row("TTFT p95 (s)",      stats_a["ttft_p95"],         stats_b["ttft_p95"])
        self.row("TTFT p99 (s)",      stats_a["ttft_p99"],         stats_b["ttft_p99"])
        self.row("TPOT mean (s/tok)", stats_a["tpot_mean"],        stats_b["tpot_mean"],  fmt=".4f")
        self.row("Throughput(tok/s)", stats_a["throughput_tok_s"], stats_b["throughput_tok_s"], higher_is_better=True, fmt=".1f")
        self.row("Requests/s",        stats_a["requests_per_s"],   stats_b["requests_per_s"],  higher_is_better=True, fmt=".2f")
        self.row("Error rate",        stats_a["error_rate"],       stats_b["error_rate"], fmt=".1%")

    def ascii_bar_chart(self, name_a, val_a, name_b, val_b, label, higher_is_better=False):
        print(f"
  {label}")
        mx = max(val_a, val_b, 0.001)
        for name, val in [(name_a, val_a), (name_b, val_b)]:
            filled = int((val / mx) * 30)
            bar    = chr(9608)*filled + chr(9617)*(30-filled)
            winner = " v" if (higher_is_better and val == max(val_a,val_b)) or (not higher_is_better and val == min(val_a,val_b)) else ""
            print(f"  {name:<10} {bar}  {val:.1f}{winner}")

    def verdict(self, name_a, stats_a, name_b, stats_b):
        sp = stats_b["throughput_tok_s"] / max(stats_a["throughput_tok_s"], 0.001)
        lp = stats_a["ttft_p50"]         / max(stats_b["ttft_p50"],         0.001)
        print()
        print("=" * self.WIDTH)
        print("  VERDICT")
        print("=" * self.WIDTH)
        print(f"  Throughput: {name_b} is {sp:.1f}x faster than {name_a}")
        direction = "faster" if lp > 1 else "slower"
        print(f"  TTFT p50:   {name_a} is {abs(lp):.1f}x {direction} than {name_b}")
        print()
        print(f"  USE {name_a:<12} for: edge/local, no GPU, privacy, offline, low concurrency")
        print(f"  USE {name_b:<12} for: cloud/API, high throughput, many concurrent users")
        print("=" * self.WIDTH)


lc_stats = {"ttft_p50":0.41,"ttft_p95":0.89,"ttft_p99":1.21,"tpot_mean":0.023,
            "tpot_p50":0.021,"throughput_tok_s":43.5,"requests_per_s":0.87,
            "total_requests":48,"total_tokens":3840,"total_time_s":88.3,
            "error_count":0,"error_rate":0.0}
vl_stats = {"ttft_p50":0.18,"ttft_p95":0.31,"ttft_p99":0.44,"tpot_mean":0.009,
            "tpot_p50":0.008,"throughput_tok_s":112.4,"requests_per_s":2.81,
            "total_requests":48,"total_tokens":3840,"total_time_s":34.2,
            "error_count":0,"error_rate":0.0}

r = BenchmarkReporter()
r.header("Inference Benchmark: llama.cpp vs vLLM")
r.comparison_table("llama.cpp", lc_stats, "vLLM", vl_stats)
r.ascii_bar_chart("llama.cpp", lc_stats["throughput_tok_s"],
                  "vLLM", vl_stats["throughput_tok_s"],
                  "Throughput (tok/s)", higher_is_better=True)
r.verdict("llama.cpp", lc_stats, "vLLM", vl_stats)

import random

class MetricsCalculator:
    @staticmethod
    def percentile(values, p):
        if not values: return 0.0
        sv = sorted(values)
        idx = (p/100.0)*(len(sv)-1)
        lo = int(idx); hi = min(lo+1, len(sv)-1)
        return sv[lo] + (idx-lo)*(sv[hi]-sv[lo])
    @staticmethod
    def compute(results):
        if not results: return {}
        calc = MetricsCalculator
        good   = [r for r in results if r.get("tokens",0)>0]
        ttfts  = [r["ttft"]       for r in good]
        tpots  = [r["tpot"]       for r in good]
        times  = [r["total_time"] for r in results]
        tokens = [r["tokens"]     for r in results]
        errors = len(results) - len(good)
        tt = sum(tokens); ti = sum(times)
        return {
            "ttft_p50":calc.percentile(ttfts,50),"ttft_p95":calc.percentile(ttfts,95),
            "ttft_p99":calc.percentile(ttfts,99),"ttft_mean":sum(ttfts)/len(ttfts) if ttfts else 0,
            "tpot_p50":calc.percentile(tpots,50),"tpot_p95":calc.percentile(tpots,95),
            "tpot_mean":sum(tpots)/len(tpots) if tpots else 0,
            "throughput_tok_s":tt/ti if ti else 0,"requests_per_s":len(results)/ti if ti else 0,
            "total_requests":len(results),"total_tokens":tt,"total_time_s":ti,
            "error_count":errors,"error_rate":errors/len(results),"avg_tokens":tt/len(results),
        }

class BenchmarkReporter:
    WIDTH = 64
    def header(self,t):
        p=(self.WIDTH-len(t)-2)//2; print("="*self.WIDTH); print(" "*p+t); print("="*self.WIDTH)
    def row(self,label,va,vb,higher=False,fmt=".3f"):
        sa=format(va,".1%" if fmt==".1%" else fmt); sb=format(vb,".1%" if fmt==".1%" else fmt)
        if va!=vb:
            if (not higher and va<vb) or (higher and va>vb): sa+=" v"
            else: sb+=" v"
        print(f"  {label:<24} {sa:>18} {sb:>18}")
    def comparison_table(self,na,sa,nb,sb):
        print(f"  {'Metric':<24} {na:>18} {nb:>18}"); print("  "+"-"*62)
        self.row("TTFT p50 (s)",sa["ttft_p50"],sb["ttft_p50"])
        self.row("TTFT p95 (s)",sa["ttft_p95"],sb["ttft_p95"])
        self.row("TTFT p99 (s)",sa["ttft_p99"],sb["ttft_p99"])
        self.row("TPOT mean(s/tok)",sa["tpot_mean"],sb["tpot_mean"],fmt=".4f")
        self.row("Throughput(tok/s)",sa["throughput_tok_s"],sb["throughput_tok_s"],higher=True,fmt=".1f")
        self.row("Error rate",sa["error_rate"],sb["error_rate"],fmt=".1%")
    def ascii_bar(self,na,va,nb,vb,label,higher=False):
        print(f"
  {label}"); mx=max(va,vb,0.001)
        for n,v in [(na,va),(nb,vb)]:
            f=int((v/mx)*30); bar=chr(9608)*f+chr(9617)*(30-f)
            w=" v" if (higher and v==max(va,vb)) or (not higher and v==min(va,vb)) else ""
            print(f"  {n:<10} {bar}  {v:.1f}{w}")
    def verdict(self,na,sa,nb,sb):
        sp=sb["throughput_tok_s"]/max(sa["throughput_tok_s"],0.001)
        lp=sa["ttft_p50"]/max(sb["ttft_p50"],0.001)
        print(); print("="*self.WIDTH); print("  VERDICT"); print("="*self.WIDTH)
        print(f"  Throughput: {nb} is {sp:.1f}x faster")
        print(f"  TTFT p50:   {'faster' if lp>1 else 'slower'} by {abs(lp):.1f}x")
        print(f"  USE {na:<12} for: local/edge, privacy, no GPU")
        print(f"  USE {nb:<12} for: cloud, high concurrency, throughput")
        print("="*self.WIDTH)

def simulate_benchmark(n_prompts=48, seed=42):
    rng = random.Random(seed)
    profiles = {
        "llamacpp":{"ttft_mean":0.40,"ttft_std":0.12,"ttft_min":0.05,
                    "tpot_mean":0.022,"tpot_std":0.005,"tpot_min":0.005,"error_rate":0.02},
        "vllm":    {"ttft_mean":0.17,"ttft_std":0.05,"ttft_min":0.02,
                    "tpot_mean":0.008,"tpot_std":0.002,"tpot_min":0.001,"error_rate":0.005},
    }
    results = {}
    for engine, p in profiles.items():
        rows = []
        for _ in range(n_prompts):
            if rng.random() < p["error_rate"]:
                rows.append({"ttft":0,"tpot":0,"total_time":0.1,"tokens":0,"tokens_per_s":0})
                continue
            ttft   = max(rng.gauss(p["ttft_mean"], p["ttft_std"]), p["ttft_min"])
            tpot   = max(rng.gauss(p["tpot_mean"], p["tpot_std"]), p["tpot_min"])
            tokens = rng.randint(40, 130)
            total  = ttft + tpot * max(tokens-1, 0)
            rows.append({"ttft":ttft,"tpot":tpot,"total_time":total,
                         "tokens":tokens,"tokens_per_s":tokens/total})
        results[engine] = rows
    return results

def run_simulated_report():
    data = simulate_benchmark()
    lc   = MetricsCalculator.compute(data["llamacpp"])
    vl   = MetricsCalculator.compute(data["vllm"])
    r    = BenchmarkReporter()
    r.header("Simulated Benchmark: llama.cpp vs vLLM")
    r.comparison_table("llama.cpp", lc, "vLLM", vl)
    r.ascii_bar("llama.cpp",lc["throughput_tok_s"],"vLLM",vl["throughput_tok_s"],
                "Throughput (tok/s)", higher=True)
    r.ascii_bar("llama.cpp",lc["ttft_p50"],"vLLM",vl["ttft_p50"],"TTFT p50 (s)")
    r.verdict("llama.cpp", lc, "vLLM", vl)
    return {"llamacpp": lc, "vllm": vl}

run_simulated_report()

import json

class ResultsAnalyzer:
    def __init__(self):
        self.data = {}

    def load(self, filepath):
        with open(filepath) as f:
            self.data = json.load(f)

    def from_dict(self, data):
        self.data = data

    def speedup_summary(self):
        lc = self.data.get("llamacpp", {})
        vl = self.data.get("vllm",     {})
        def safe_div(a, b): return round(a / b, 2) if b and b != 0 else 0.0
        tps_sp   = safe_div(vl.get("throughput_tok_s",1), lc.get("throughput_tok_s",1))
        ttft_sp  = safe_div(lc.get("ttft_p50",1),         vl.get("ttft_p50",1))
        tpot_sp  = safe_div(lc.get("tpot_mean",1),        vl.get("tpot_mean",1))
        return {
            "throughput_speedup":  tps_sp,
            "ttft_p50_speedup":    ttft_sp,
            "tpot_speedup":        tpot_sp,
            "winner_throughput":   "vllm" if tps_sp >= 1 else "llamacpp",
            "winner_latency":      "vllm" if ttft_sp >= 1 else "llamacpp",
        }

    def recommendations(self, use_case):
        s  = self.speedup_summary()
        sp = s["throughput_speedup"]
        lp = s["ttft_p50_speedup"]
        recs = {
            "chatbot": [
                f"vLLM is recommended — {lp:.1f}x lower TTFT p50 means snappier responses.",
                "Enable prefix caching for shared system prompts to further cut latency.",
                "Use speculative decoding (--speculative-model [ngram]) for another 1.5-3x latency gain.",
                "llama.cpp is viable for single-user local chatbots with no GPU available.",
            ],
            "batch_processing": [
                f"vLLM's {sp:.1f}x throughput advantage makes it the clear choice for batch jobs.",
                "Use vllm.LLM offline mode (not the server) for maximum batch throughput.",
                "Set temperature=0.0 and fixed seeds for reproducible batch evaluation runs.",
                "llama.cpp is suitable only if you have no GPU and batches are small (<100 items).",
            ],
            "edge": [
                "llama.cpp is the only viable option for edge/embedded deployment.",
                "Use Q4_K_M quantization to fit 7B models in 4-6 GB RAM.",
                "CPU-only inference with llama.cpp is production-ready for low-traffic edge nodes.",
                "Consider smaller models (3B) for latency-sensitive edge applications.",
            ],
            "rag": [
                f"vLLM with prefix caching eliminates redundant KV compute on shared context.",
                "Enable --enable-prefix-caching to cache retrieved document embeddings.",
                f"vLLM handles {sp:.1f}x more concurrent RAG requests than llama.cpp.",
                "Use a dedicated embedding model server alongside vLLM for the retrieval step.",
            ],
            "api_service": [
                f"vLLM is strongly recommended — {sp:.1f}x throughput and proper concurrent request handling.",
                "Deploy behind nginx with least_conn load balancing for multiple GPU replicas.",
                "Set --max-num-seqs based on your SLA: lower = better latency, higher = better throughput.",
                "Monitor vllm:num_requests_waiting — scale horizontally when queue depth exceeds 10.",
            ],
        }
        return recs.get(use_case, [f"Unknown use_case '{use_case}'. Choose from: {list(recs)}"])

    def export_markdown(self, filepath):
        lc = self.data.get("llamacpp", {})
        vl = self.data.get("vllm",     {})
        s  = self.speedup_summary()
        lines = [
            "# Benchmark Report: llama.cpp vs vLLM
",
            "## Metrics Comparison
",
            "| Metric | llama.cpp | vLLM |",
            "|--------|-----------|------|",
        ]
        for key, label in [("ttft_p50","TTFT p50 (s)"),("ttft_p95","TTFT p95 (s)"),
                            ("tpot_mean","TPOT mean (s/tok)"),("throughput_tok_s","Throughput (tok/s)")]:
            lines.append(f"| {label} | {lc.get(key,0):.3f} | {vl.get(key,0):.3f} |")
        lines += ["
## Speedup Summary
",
                  f"- Throughput: vLLM is **{s['throughput_speedup']:.1f}x** faster",
                  f"- TTFT p50:   vLLM is **{s['ttft_p50_speedup']:.1f}x** lower latency
",
                  "## Recommendations
"]
        for uc in ["chatbot","batch_processing","api_service"]:
            lines.append(f"### {uc.replace('_',' ').title()}")
            for r in self.recommendations(uc):
                lines.append(f"- {r}")
            lines.append("")
        with open(filepath, "w") as f:
            f.write("
".join(lines))
        print(f"Markdown report saved to {filepath}")

    def regression_check(self, baseline_filepath):
        with open(baseline_filepath) as f:
            baseline = json.load(f)
        warnings = []
        for engine in ("llamacpp", "vllm"):
            curr = self.data.get(engine, {})
            base = baseline.get(engine, {})
            for metric in ("ttft_p50","ttft_p95","tpot_mean"):
                c, b = curr.get(metric,0), base.get(metric,0)
                if b > 0 and c > b * 1.10:
                    warnings.append(f"[{engine}] {metric} regressed: {b:.3f} → {c:.3f} (+{(c/b-1)*100:.1f}%)")
            for metric in ("throughput_tok_s","requests_per_s"):
                c, b = curr.get(metric,0), base.get(metric,0)
                if b > 0 and c < b * 0.90:
                    warnings.append(f"[{engine}] {metric} regressed: {b:.1f} → {c:.1f} ({(c/b-1)*100:.1f}%)")
        return warnings


mock_data = {
    "llamacpp":{"ttft_p50":0.41,"ttft_p95":0.89,"ttft_p99":1.21,
                "tpot_mean":0.023,"throughput_tok_s":43.5,
                "requests_per_s":0.87,"error_rate":0.0,"total_tokens":3840},
    "vllm":    {"ttft_p50":0.18,"ttft_p95":0.31,"ttft_p99":0.44,
                "tpot_mean":0.009,"throughput_tok_s":112.4,
                "requests_per_s":2.81,"error_rate":0.0,"total_tokens":3840},
}

analyzer = ResultsAnalyzer()
analyzer.from_dict(mock_data)

summary = analyzer.speedup_summary()
print("Speedup Summary:")
for k, v in summary.items():
    print(f"  {k:<28} {v}")

print("
Recommendations for chatbot:")
for r in analyzer.recommendations("chatbot"):
    print(f"  - {r}")

print("
Recommendations for batch_processing:")
for r in analyzer.recommendations("batch_processing"):
    print(f"  - {r}")