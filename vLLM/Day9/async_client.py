"""
Day 9 — Async concurrent requests to vLLM
vLLM's server handles many requests simultaneously via continuous batching.
This file shows how to fire multiple requests concurrently from the client side
using asyncio + the async OpenAI client.
"""

import asyncio
import time
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


async def single_request(prompt: str, req_id: int) -> dict:
    """Send one chat request and return timing info."""
    t0 = time.time()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.7,
    )
    elapsed = time.time() - t0
    return {
        "id":     req_id,
        "prompt": prompt[:40],
        "reply":  response.choices[0].message.content[:60],
        "tokens": response.usage.total_tokens,
        "time":   round(elapsed, 3),
    }


async def run_concurrent(prompts: list[str]) -> None:
    """
    Fire all prompts concurrently.
    vLLM batches them together server-side via continuous batching —
    total wall time is much less than sequential sum.
    """
    t_start = time.time()

    tasks   = [single_request(p, i + 1) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    wall_time = time.time() - t_start

    print(f"\n{'═' * 56}")
    print(f"  Concurrent Requests: {len(prompts)}  |  Wall time: {wall_time:.2f}s")
    print(f"{'═' * 56}")
    for r in results:
        print(f"  [{r['id']}] {r['time']}s | {r['tokens']} tok | {r['prompt']}...")
    print(f"{'─' * 56}")
    print(f"  Effective throughput: {len(prompts) / wall_time:.1f} req/s")
    print(f"{'═' * 56}")


async def sequential_vs_concurrent(prompts: list[str]) -> None:
    """Compare sequential vs concurrent request timing."""

    # Sequential
    t0 = time.time()
    for i, p in enumerate(prompts):
        await single_request(p, i)
    seq_time = time.time() - t0

    # Concurrent
    t0 = time.time()
    tasks = [single_request(p, i) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)
    con_time = time.time() - t0

    print(f"\n  Sequential : {seq_time:.2f}s")
    print(f"  Concurrent : {con_time:.2f}s")
    print(f"  Speedup    : {seq_time / con_time:.1f}×")


if __name__ == "__main__":
    prompts = [
        "What is PagedAttention?",
        "Explain continuous batching in one sentence.",
        "What is the KV cache?",
        "Why is vLLM faster than HuggingFace?",
        "What does temperature do in sampling?",
    ]

    print("── Concurrent requests ─────────────────────────")
    asyncio.run(run_concurrent(prompts))

    print("\n── Sequential vs Concurrent ────────────────────")
    asyncio.run(sequential_vs_concurrent(prompts[:3]))
