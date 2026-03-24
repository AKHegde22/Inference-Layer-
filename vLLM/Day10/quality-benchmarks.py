from openai import OpenAI

def run_quality_benchmark(model_configs, test_prompts, port_base=8000):
    results = []
    reference_responses = None

    for i, config in enumerate(model_configs):
        client = OpenAI(
            base_url=f"http://localhost:{config['port']}/v1",
            api_key="none"
        )
        responses = []
        for prompt in test_prompts:
            try:
                resp = client.chat.completions.create(
                    model="local",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                responses.append(resp.choices[0].message.content)
            except Exception as e:
                responses.append(f"[ERROR: {e}]")

        if i == 0:
            reference_responses = responses
            score = 1.0
        else:
            matches = 0
            for ref, resp in zip(reference_responses, responses):
                ref_words  = set(ref.lower().split())
                resp_words = set(resp.lower().split())
                if len(ref_words & resp_words) >= 3:
                    matches += 1
            score = matches / len(test_prompts) if test_prompts else 0.0

        results.append({
            "name":               config["name"],
            "quantization":       config["quantization"],
            "responses":          responses,
            "consistency_score":  round(score, 2),
        })

    return results

def print_benchmark_report(results):
    sorted_r = sorted(results, key=lambda x: x["consistency_score"], reverse=True)
    print("=" * 54)
    print("  Quantization Quality Benchmark")
    print("=" * 54)
    print(f"  {'Model':<22} {'Format':<14} {'Consistency':>10}")
    print("  " + "-" * 48)
    for r in sorted_r:
        bar = "█" * int(r["consistency_score"] * 10)
        print(f"  {r['name']:<22} {r['quantization']:<14} {r['consistency_score']:>9.0%}  {bar}")
    print("=" * 54)


configs = [
    {"name": "FP16 (Reference)", "quantization": "fp16",         "port": 8001},
    {"name": "AWQ INT4",         "quantization": "awq",           "port": 8002},
    {"name": "GPTQ INT4",        "quantization": "gptq",          "port": 8003},
    {"name": "BitsAndBytes",     "quantization": "bitsandbytes",  "port": 8004},
]
prompts = [
    "What is the capital of France?",
    "Explain what a neural network is in one sentence.",
    "What is 17 multiplied by 13?",
]
results = run_quality_benchmark(configs, prompts)
print_benchmark_report(results)