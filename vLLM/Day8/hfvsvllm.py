import time

def benchmark_hf(prompts, model_name="gpt2"):
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model_name,
                    max_new_tokens=50, device=0)
    t0 = time.time()
    _ = pipe(prompts, batch_size=8)
    total = time.time() - t0
    return {
        "framework":   "HuggingFace",
        "total_time":  round(total, 3),
        "throughput":  round(len(prompts) / total, 2),
        "avg_latency": round(total / len(prompts), 3),
        "n_prompts":   len(prompts),
    }

def benchmark_vllm(prompts, model_name="gpt2"):
    from vllm import LLM, SamplingParams
    llm    = LLM(model=model_name)
    params = SamplingParams(max_tokens=50, temperature=0.0)
    t0 = time.time()
    _  = llm.generate(prompts, params)
    total = time.time() - t0
    return {
        "framework":   "vLLM",
        "total_time":  round(total, 3),
        "throughput":  round(len(prompts) / total, 2),
        "avg_latency": round(total / len(prompts), 3),
        "n_prompts":   len(prompts),
    }

def print_comparison(hf_result, vllm_result):
    speedup = vllm_result["throughput"] / hf_result["throughput"]
    print("\n" + "═" * 56)
    print(f"  Batch Inference Benchmark  ({hf_result['n_prompts']} prompts)")
    print("═" * 56)
    print(f"  {'Metric':<20} {'HuggingFace':>14} {'vLLM':>14}")
    print("─" * 56)
    for key, label in [("total_time","Total Time (s)"),
                       ("throughput","Throughput (req/s)"),
                       ("avg_latency","Avg Latency (s)")]:
        print(f"  {label:<20} {str(hf_result[key]):>14} {str(vllm_result[key]):>14}")
    print("─" * 56)
    print(f"  {'vLLM Speedup':<20} {speedup:>27.1f}×")
    print("═" * 56)


prompts = [f"The capital of country number {i} is" for i in range(20)]
hf  = benchmark_hf(prompts)
vl  = benchmark_vllm(prompts)
print_comparison(hf, vl)