"""
Day 9 — Streaming responses from vLLM
Token-by-token streaming using Server-Sent Events (SSE).
Works identically to the OpenAI streaming API.
"""

import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def stream_chat(user_message: str, system: str = "You are a helpful assistant.") -> str:
    """
    Streams tokens as they are generated.
    Each chunk contains a delta — the new piece of text since the last chunk.
    Prints tokens in real time and returns the full assembled response.
    """
    full_response = ""
    token_count   = 0
    t_start       = time.time()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=300,
        temperature=0.7,
        stream=True,          # ← this is the only difference from non-streaming
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_response += delta
            token_count   += 1

    elapsed = time.time() - t_start
    tok_per_sec = token_count / elapsed if elapsed > 0 else 0
    print(f"\n\n  [{token_count} tokens | {elapsed:.2f}s | {tok_per_sec:.1f} tok/s]")

    return full_response


def stream_with_first_token_latency(user_message: str) -> None:
    """
    Measures time-to-first-token (TTFT) separately from total generation time.
    TTFT is the latency the user perceives before anything appears on screen.
    """
    t_start      = time.time()
    first_token  = True
    ttft         = None
    token_count  = 0

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=150,
        temperature=0.0,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token:
                ttft = time.time() - t_start
                first_token = False
            print(delta, end="", flush=True)
            token_count += 1

    total = time.time() - t_start
    print(f"\n\n  TTFT        : {ttft:.3f}s")
    print(f"  Total time  : {total:.3f}s")
    print(f"  Tokens      : {token_count}")
    print(f"  Throughput  : {token_count / total:.1f} tok/s")


if __name__ == "__main__":
    print("── Streaming chat ──────────────────────────────")
    stream_chat("Explain PagedAttention in 3 sentences.")

    print("\n── First token latency ─────────────────────────")
    stream_with_first_token_latency("What is vLLM?")
