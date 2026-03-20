# Day 4: Running llama-server and Using the OpenAI-Compatible API

## Why Use llama-server Instead of llama-cli?

`llama-cli` is great for quick one-off tests, but for real use you want a persistent server.

| Feature | llama-cli | llama-server |
|---|---|---|
| Lifecycle | One request, then exits | Stays running, handles many |
| API | None | OpenAI-compatible REST API |
| Streaming | No | Full streaming support |
| Concurrency | No | Queue-based concurrency |
| Clients | CLI only | Any HTTP client or SDK |

`llama-server` is a drop-in replacement for OpenAI's API, meaning any tool built for OpenAI works with your local model:
- Python `openai` SDK
- LangChain / LlamaIndex
- Continue.dev (VS Code AI coding)
- Open-WebUI (ChatGPT-like interface)
- `curl`, Postman, any HTTP client

### Endpoints exposed by llama-server:
```
POST /v1/chat/completions    ← main chat endpoint
POST /v1/completions         ← raw text completion
POST /v1/embeddings          ← vector embeddings
GET  /v1/models              ← list available models
GET  /health                 ← server health check
GET  /metrics                ← performance metrics
```

---

## Starting llama-server

### Basic startup:
```bash
./build/bin/llama-server \
  -m models/mistral-7b-instruct.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080
```

### Key flags:
| Flag | Description |
|---|---|
| `-m` | Path to model file |
| `--host` | Bind address (`0.0.0.0` = accessible on network) |
| `--port` | Port number (default: `8080`) |
| `-c` | Context size (e.g. `-c 8192`) |
| `-ngl` | GPU layers to offload (e.g. `-ngl 35`) |
| `-t` | CPU threads |
| `--parallel` | Number of parallel request slots (default: `1`) |
| `--cont-batching` | Enable continuous batching |
| `--flash-attn` | Enable flash attention (faster, less memory) |
| `-a` | Model alias shown in `/v1/models` response |

### Examples:
```bash
# CPU only, 8 threads
./build/bin/llama-server -m model.gguf -t 8 --port 8080

# Full GPU offload + flash attention
./build/bin/llama-server -m model.gguf -ngl 99 --flash-attn

# Production: parallel requests + continuous batching
./build/bin/llama-server \
  -m model.gguf -ngl 99 \
  --parallel 4 \
  --cont-batching \
  --ctx-size 8192 \
  -a "my-model"
```

When ready, you'll see:
```
llama server listening at http://0.0.0.0:8080
```

> See `server-start-comm.bash` for a ready-to-use startup command with Llama 3 8B.

---

## Using the OpenAI-Compatible API

### With curl:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user",   "content": "What is Python?"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

### With Python `openai` SDK:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none"   # required by SDK, ignored by server
)

response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user",   "content": "Explain list comprehensions."}
    ],
    temperature=0.7,
    max_tokens=300
)
print(response.choices[0].message.content)
```

---

## Streaming Responses (`stream.py`)

Instead of waiting for the full response, streaming sends tokens as they are generated — just like ChatGPT's typing effect.

```python
from openai import OpenAI

def stream_chat(prompt, system="You are helpful.", port=8080):
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
    stream = client.chat.completions.create(
        model="local",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
        stream=True
    )
    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_response += delta
    print()
    return full_response
```

Key difference: `stream=True` returns an iterator of chunks instead of a single response object. Each chunk has a `delta.content` field with the new token(s).

---

## Multi-Turn Chat (`multi-turn.py`)

LLMs have no built-in memory between calls. To maintain a conversation, you must send the full message history on every request.

### Pattern — History Management:
```python
history = [{"role": "system", "content": "You are a helpful assistant."}]

def chat(user_input):
    history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="local",
        messages=history,   # ← send FULL history each time
        temperature=0.7,
        max_tokens=500
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply
```

### Context Window Warning:
Every turn adds tokens. Eventually you'll hit the model's context limit.

**Sliding window strategy** — keep the system prompt and only the last N exchanges:
```python
def _trim_history(self):
    max_msgs = self.max_history_turns * 2
    if len(self.history) - 1 > max_msgs:
        self.history = [self.history[0]] + self.history[-max_msgs:]
```

The `ChatBot` class in `multi-turn.py` wraps all of this with a configurable `max_history_turns` parameter and a `reset()` method.

### Monitoring token usage:
```python
usage = response.usage
print(f"Prompt tokens:     {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total:             {usage.total_tokens}")
```

---

## Benchmarking the Server (`api-health.py`)

`api-health.py` does two things: verifies the server is reachable, then benchmarks response time and throughput across a set of prompts.

```
Prompt                         |  Avg Time | Avg Tokens | Tokens/sec
--------------------------------------------------------------------
What is Python?                |     1.23s |         45 |       36.6
Write a bubble sort in Python  |     3.87s |        112 |       28.9
Explain transformers in one... |     2.54s |         89 |       35.0
```

It hits `/health` first — if the server isn't up, it raises a `ConnectionError` immediately rather than hanging on a completion request.

```python
r = requests.get(f"http://localhost:{port}/health", timeout=3)
r.raise_for_status()
```

Run it to get a quick baseline of your model's performance before building on top of it.
