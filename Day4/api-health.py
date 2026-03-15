import time
import requests
from openai import OpenAI

def benchmark_server(port, prompts, n_runs=1):
    # Health check
    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=3)
        r.raise_for_status()
        print("Server OK\n")
    except Exception:
        raise ConnectionError(f"Server not reachable at port {port}")

    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")

    print(f"{'Prompt':<30} | {'Avg Time':>9} | {'Avg Tokens':>10} | {'Tokens/sec':>10}")
    print("-" * 68)

    for prompt in prompts:
        times, token_counts = [], []
        for _ in range(n_runs):
            t0 = time.time()
            resp = client.chat.completions.create(
                model="local",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0
            )
            elapsed = time.time() - t0
            content = resp.choices[0].message.content or ""
            tokens = len(content.split())
            times.append(elapsed)
            token_counts.append(tokens)

        avg_time = sum(times) / len(times)
        avg_tok  = sum(token_counts) / len(token_counts)
        tps = avg_tok / avg_time if avg_time > 0 else 0
        label = prompt[:28] + ".." if len(prompt) > 30 else prompt
        print(f"{label:<30} | {avg_time:>8.2f}s | {avg_tok:>10.0f} | {tps:>9.1f}")

prompts = [
    "What is Python?",
    "Write a bubble sort in Python",
    "Explain transformers in one paragraph"
]
benchmark_server(8080, prompts, n_runs=2)