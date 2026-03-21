# Day 9: vLLM OpenAI-Compatible Server

Day 8 covered the internals — PagedAttention, continuous batching, the scheduler. Day 9 is about using vLLM in practice: starting the server, hitting it with the OpenAI API, streaming tokens, managing multi-turn history, and firing concurrent requests with asyncio.

The vLLM server exposes the exact same API as OpenAI. Any code written for `api.openai.com` works against `localhost:8000` with one line changed.

---

## File Structure

```
Day9/
├── server_start.bash  ← How to start the vLLM server (with all flag variants)
├── api_health.py      ← Health check and model info endpoints
├── completions.py     ← /v1/completions and /v1/chat/completions
├── streaming.py       ← Token-by-token streaming + TTFT measurement
├── multi_turn.py      ← Stateful multi-turn chat session class
└── async_client.py    ← Concurrent requests with asyncio
```

---

## Setup

```bash
pip install vllm openai
```

Start the server before running any of the Python files:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000
```

The server is ready when you see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## server_start.bash — Starting the Server

Documents all the common server launch configurations in one place.

```bash
# Basic
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000

# With memory tuning
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096

# Multi-GPU (2 GPUs)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --tensor-parallel-size 2

# Quantized (AWQ)
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
  --quantization awq
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model ID or local path |
| `--port` | 8000 | Port to listen on |
| `--gpu-memory-utilization` | 0.90 | Fraction of VRAM for KV cache |
| `--max-model-len` | model default | Override max context length |
| `--tensor-parallel-size` | 1 | Number of GPUs to split across |
| `--quantization` | None | `awq`, `gptq`, `fp8` |
| `--max-num-seqs` | 256 | Max concurrent sequences |
| `--served-model-name` | model id | Alias in `/v1/models` response |

---

## api_health.py — Health Check

```bash
python api_health.py
```

```
════════════════════════════════════════
  vLLM Server Health Check
════════════════════════════════════════
  Health:  OK

  Loaded models (1):
    id         : mistralai/Mistral-7B-Instruct-v0.2
    object     : model
    created    : 1710000000

  Model ID        : mistralai/Mistral-7B-Instruct-v0.2
  Max model len   : 4096
  Quantization    : none
```

Two endpoints used:
- `GET /health` — returns 200 when ready, used for readiness probes
- `GET /v1/models` — lists loaded models with vLLM-specific metadata

---

## completions.py — Raw and Chat Completions

Covers both completion styles the server supports.

```bash
python completions.py
```

### /v1/completions — raw text

Continues raw text with no chat template. Good for fill-in-the-middle or custom prompt formats.

```python
response = client.completions.create(
    model=MODEL,
    prompt="The three laws of robotics are",
    max_tokens=100,
    temperature=0.7,
)
print(response.choices[0].text)
```

### /v1/chat/completions — chat format

Applies the model's chat template automatically. The standard way to use instruction-tuned models.

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is the KV cache?"},
    ],
    max_tokens=200,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

### Response metadata

```
  Reply         : The KV cache stores key and value vectors...
  Finish reason : stop
  Prompt tokens : 28
  Output tokens : 94
  Total tokens  : 122
```

`finish_reason` is `"stop"` when the model hit a stop token, `"length"` when it hit `max_tokens`.

---

## streaming.py — Token-by-Token Streaming

```bash
python streaming.py
```

The only difference from non-streaming is `stream=True`. Each chunk contains a `delta` — the new text since the last chunk.

```python
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Explain PagedAttention."}],
    max_tokens=300,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### Output

```
PagedAttention is a memory management technique that applies the concept
of virtual memory paging to the KV cache in LLM inference...

  [87 tokens | 1.43s | 60.8 tok/s]
```

### Time-to-first-token (TTFT)

TTFT is the latency the user perceives before anything appears. The file measures it separately:

```
  TTFT        : 0.082s
  Total time  : 1.430s
  Tokens      : 87
  Throughput  : 60.8 tok/s
```

TTFT is dominated by the prefill phase (processing the prompt). Generation throughput is measured after that.

---

## multi_turn.py — Multi-Turn Chat Session

```bash
python multi_turn.py
```

The vLLM server is stateless — it doesn't remember previous turns. The client maintains history and sends the full context every call. This is identical to how the OpenAI API works.

```python
session = VLLMChat(system="You are a concise AI tutor.")

session.chat("What is a transformer model?")
session.chat("What is the attention mechanism?")
session.chat("How does that relate to the KV cache?")

session.show_history()
```

### History trimming

When history exceeds `max_history` turns, the oldest pairs are dropped. The system prompt is always preserved.

```
max_history = 20  →  keeps last 40 messages + system prompt
```

### Output

```
User: What is a transformer model?
Assistant: A transformer is a neural network architecture that uses
self-attention to process sequences in parallel...

User: What is the attention mechanism?
Assistant: Attention computes a weighted sum of value vectors...

User: How does that relate to the KV cache?
Assistant: The KV cache stores the key and value vectors from previous
tokens so they don't need to be recomputed each step...

  History (6 messages | ~380 tokens used):
  [1] USER: What is a transformer model?...
  [2] ASSISTANT: A transformer is a neural network architecture...
  [3] USER: What is the attention mechanism?...
  [4] ASSISTANT: Attention computes a weighted sum...
  [5] USER: How does that relate to the KV cache?...
  [6] ASSISTANT: The KV cache stores the key and value vectors...
```

---

## async_client.py — Concurrent Requests

```bash
python async_client.py
```

Uses `AsyncOpenAI` + `asyncio.gather()` to fire multiple requests simultaneously. vLLM batches them together server-side via continuous batching.

```python
async def run_concurrent(prompts):
    tasks   = [single_request(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
```

### Output

```
════════════════════════════════════════════════════════
  Concurrent Requests: 5  |  Wall time: 1.84s
════════════════════════════════════════════════════════
  [1] 1.71s | 112 tok | What is PagedAttention?...
  [2] 1.68s | 89 tok  | Explain continuous batching...
  [3] 1.74s | 95 tok  | What is the KV cache?...
  [4] 1.80s | 103 tok | Why is vLLM faster than HuggingFace?...
  [5] 1.84s | 78 tok  | What does temperature do in sampling?...
────────────────────────────────────────────────────────
  Effective throughput: 2.7 req/s
════════════════════════════════════════════════════════

  Sequential : 7.21s
  Concurrent : 1.84s
  Speedup    : 3.9×
```

All 5 requests run in roughly the same time as 1 — because vLLM processes them in the same batch.

---

## How the Server API Maps to vLLM Internals

```
Client request
    │
    ▼
POST /v1/chat/completions
    │
    ▼
AsyncLLMEngine          ← applies chat template, tokenizes
    │
    ▼
Scheduler               ← adds to waiting queue
    │
    ▼
Continuous batching     ← slots in at next available step
    │
    ▼
PagedAttention          ← allocates KV blocks on demand
    │
    ▼
Token generation        ← streamed back via SSE
    │
    ▼
Client receives chunks
```

---

## Day 8 vs Day 9

| Day 8 | Day 9 |
|---|---|
| Why vLLM exists | Using vLLM in practice |
| PagedAttention theory | OpenAI-compatible server |
| Simulated scheduler (Python) | Real server, real requests |
| SamplingParams presets | Streaming, multi-turn, async |
| No GPU needed for most files | Requires GPU + running server |
