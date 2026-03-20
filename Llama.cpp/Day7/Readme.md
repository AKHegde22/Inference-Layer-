# Day 7: Capstone Project — LlamaCLI Chatbot

Build a full-featured terminal chatbot that ties together everything from Days 1–6: quantization, sampling parameters, chat templates, the OpenAI-compatible API, GPU tuning, and structured output.

```
┌──────────────────────────────────────────────────────┐
│  🦙 LlamaCLI  —  Local AI Chat                       │
│  Host:  localhost:8080  |  CTX: 4096                 │
├──────────────────────────────────────────────────────┤
│                                                      │
│  You: Explain list comprehensions in Python          │
│                                                      │
│  Assistant: List comprehensions are a concise way... │
│                                                      │
│  [87 tokens | 1.23s | 70.7 tok/s]                   │
├──────────────────────────────────────────────────────┤
│  > _                                                 │
└──────────────────────────────────────────────────────┘
```

---

## File Structure

```
Day7/
├── main.py       ← Entry point, REPL loop
├── client.py     ← LlamaClient (OpenAI SDK wrapper + streaming)
├── session.py    ← ChatSession (history, trimming, save/load)
├── commands.py   ← /command handlers
├── display.py    ← Terminal formatting helpers
└── config.py     ← Default settings & constants
```

---

## Setup

```bash
pip install openai requests
```

Start llama-server first (from your llama.cpp build):

```bash
./build/bin/llama-server \
  -m models/mistral.Q4_K_M.gguf \
  -ngl 99 \
  --port 8080
```

Then run the chatbot:

```bash
cd Day7
python main.py

# Optional flags
python main.py --port 8080 --system "You are a Python tutor." --model mistral-7b
```

---

## Architecture

```
┌─────────────┐    HTTP/SSE    ┌──────────────────┐
│  LlamaCLI   │ ─────────────▶ │  llama-server    │
│  (Python)   │ ◀───────────── │  (local process) │
└─────────────┘    Streaming   └──────────────────┘
       │
       ├── ChatSession   (history, token tracking, save/load)
       ├── LlamaClient   (OpenAI SDK wrapper, streaming, bench)
       ├── handle_command (slash commands)
       └── Display        (formatted terminal output)
```

---

## Commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/system <text>` | Change the system prompt |
| `/clear` | Clear conversation history |
| `/history` | Show message history with token counts |
| `/save <file>` | Save conversation to JSON (default: `chat.json`) |
| `/load <file>` | Load conversation from JSON |
| `/params` | Show current sampling parameters |
| `/set <key> <val>` | Change a parameter (see below) |
| `/bench` | Run a quick 3-prompt benchmark |
| `/exit` | Quit |

### Settable parameters via `/set`:

| Key | Type | Default | Effect |
|---|---|---|---|
| `temperature` | float | 0.7 | Randomness of output |
| `top_p` | float | 0.9 | Nucleus sampling cutoff |
| `top_k` | int | 40 | Top-K sampling |
| `max_tokens` | int | 512 | Max tokens per response |
| `repeat_penalty` | float | 1.1 | Penalise repeated tokens |

```
You: /set temperature 0.2
  Set temperature = 0.2

You: /set max_tokens 1024
  Set max_tokens = 1024
```

---

## ChatSession — How History Works

The session stores every message as a dict with `role`, `content`, and `tokens`. Before each API call, `api_messages()` strips the internal `tokens` field since the OpenAI spec only accepts `role` and `content`.

```python
session.add("user", "What is Python?")
session.add("assistant", "Python is a programming language.", 32)

session.api_messages()
# [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
```

### Sliding window trimming

When the conversation exceeds `max_history * 2` non-system messages, the oldest messages are dropped while the system prompt is always preserved:

```
max_history = 20  →  keeps last 40 messages + system prompt
```

This prevents context window overflow without losing the model's persona.

### Save and load

```
You: /save my_chat.json
  Saved 9 messages to my_chat.json

You: /load my_chat.json
  Loaded 9 messages from my_chat.json (saved at 2025-03-19T14:32:01)
```

The JSON format:

```json
{
  "version": "1.0",
  "system_prompt": "You are a helpful AI assistant.",
  "messages": [...],
  "token_count": 312,
  "turn_count": 4,
  "saved_at": "2025-03-19T14:32:01.123456"
}
```

---

## Streaming

Responses stream token-by-token using the OpenAI SDK's `stream=True`. Each chunk is printed immediately as it arrives — the user sees output in real time rather than waiting for the full response.

```python
stream = client.chat.completions.create(..., stream=True)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

After the stream completes, per-response stats are shown:

```
[87 tokens | 1.23s | 70.7 tok/s]
```

---

## Quick Benchmark (`/bench`)

Sends 3 short prompts and reports average latency and throughput:

```
You: /bench
  Running quick benchmark (3 prompts)...

  Assistant: 4
  Assistant: Blue
  Assistant: Hi there!

  Avg latency: 0.84s
  Avg tok/s:   62.3
```

---

## config.py Reference

```python
DEFAULT_CONFIG = {
    "host":           "localhost",
    "port":           8080,
    "model":          "local",
    "temperature":    0.7,
    "top_p":          0.9,
    "top_k":          40,
    "repeat_penalty": 1.1,
    "max_tokens":     512,
    "context_limit":  4096,
    "max_history":    20,
    "system_prompt":  "You are a helpful AI assistant.",
}
```

---

## Week 1 Concepts Applied

| Day | Concept used in LlamaCLI |
|---|---|
| Day 1 | llama-server process, GGUF model loading |
| Day 2 | Q4_K_M quantization for the model file |
| Day 3 | Temperature, top_p, repeat_penalty via `/set` |
| Day 4 | OpenAI-compatible API, streaming SSE, multi-turn history |
| Day 5 | `-ngl 99` GPU offload, `--flash-attn` for the server |
| Day 6 | Health check endpoint, JSON-structured save/load |
