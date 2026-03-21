"""
Day 9 — vLLM OpenAI-compatible completions
Covers: /v1/completions (raw text) and /v1/chat/completions (chat format).
Both endpoints work identically to the OpenAI API.
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",          # vLLM doesn't require a real key
)

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # must match --model on server


# ── 1. Raw text completion (/v1/completions) ──────────────────────────────────

def raw_completion(prompt: str, max_tokens: int = 100) -> str:
    """
    /v1/completions — continues raw text, no chat template applied.
    Useful for completion tasks, fill-in-the-middle, or custom prompting.
    """
    response = client.completions.create(
        model=MODEL,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].text


# ── 2. Chat completion (/v1/chat/completions) ─────────────────────────────────

def chat_completion(user_message: str, system: str = "You are a helpful assistant.") -> str:
    """
    /v1/chat/completions — applies the model's chat template automatically.
    This is the standard way to interact with instruction-tuned models.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user_message},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message.content


# ── 3. Inspect the full response object ──────────────────────────────────────

def chat_with_metadata(user_message: str) -> None:
    """Shows the full response object — usage stats, finish reason, model info."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=100,
        temperature=0.0,
    )
    choice = response.choices[0]
    usage  = response.usage

    print(f"  Reply         : {choice.message.content}")
    print(f"  Finish reason : {choice.finish_reason}")   # "stop" or "length"
    print(f"  Prompt tokens : {usage.prompt_tokens}")
    print(f"  Output tokens : {usage.completion_tokens}")
    print(f"  Total tokens  : {usage.total_tokens}")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Raw completion ──────────────────────────────")
    text = raw_completion("The three laws of robotics are")
    print(text)

    print("\n── Chat completion ─────────────────────────────")
    reply = chat_completion("What is the KV cache in transformers?")
    print(reply)

    print("\n── Full response metadata ──────────────────────")
    chat_with_metadata("What is 2 + 2?")
