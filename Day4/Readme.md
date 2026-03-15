# Why Use llama-server?

```
llama-cli is great for quick tests, but for real use you want a SERVER.

  llama-cli                    llama-server
  ──────────────────────────   ──────────────────────────
  One request, then exits      Stays running, handles many
  No API                       OpenAI-compatible REST API
  No streaming                 Full streaming support
  No concurrency               Queue-based concurrency
  CLI only                     Any HTTP client / SDK

llama-server gives you a drop-in replacement for OpenAI's API.
This means ALL tools that work with OpenAI work with your local model:
  • Python openai SDK
  • LangChain / LlamaIndex
  • Continue.dev (VS Code AI coding)
  • Open-WebUI (ChatGPT-like interface)
  • curl, Postman, any HTTP client

ENDPOINTS exposed by llama-server:
  POST /v1/chat/completions    ← Main chat endpoint
  POST /v1/completions         ← Raw text completion
  POST /v1/embeddings          ← Vector embeddings
  GET  /v1/models              ← List available models
  GET  /health                 ← Server health check
  GET  /metrics                ← Performance metrics
```


# Starting llama-server

```
Basic startup:
  ./build/bin/llama-server \
    -m models/mistral-7b-instruct.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080

Key server flags:
  -m    model path
  --host        bind address (0.0.0.0 = accessible on network)
  --port        port number (default: 8080)
  -c            context size (e.g. -c 8192)
  -ngl          GPU layers to offload (e.g. -ngl 35)
  -t            CPU threads
  --parallel    number of parallel request slots (default: 1)
  --cont-batching   enable continuous batching
  --flash-attn      enable flash attention (faster, less memory)
  -a            model alias shown in /v1/models response

EXAMPLES:

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

  # You'll see this when ready:
  # llama server listening at http://0.0.0.0:8080
```


# Using the OpenAI-Compatible API

```
Once the server is running, query it with any HTTP client.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WITH CURL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WITH PYTHON openai SDK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  from openai import OpenAI

  client = OpenAI(
      base_url="http://localhost:8080/v1",
      api_key="none"          # required by SDK, ignored by server
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STREAMING RESPONSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  stream = client.chat.completions.create(
      model="local",
      messages=[{"role": "user", "content": "Write a haiku"}],
      stream=True
  )
  for chunk in stream:
      delta = chunk.choices[0].delta.content or ""
      print(delta, end="", flush=True)
```


# Building a Multi-Turn Chat App

```
For real conversations, you must maintain message history yourself.
llama.cpp (and all LLMs) have NO built-in memory between calls.

PATTERN — History Management:

  history = [
      {"role": "system", "content": "You are a helpful assistant."}
  ]

  def chat(user_input):
      history.append({"role": "user", "content": user_input})
  
      response = client.chat.completions.create(
          model="local",
          messages=history,        # ← send FULL history each time
          temperature=0.7,
          max_tokens=500
      )
  
      reply = response.choices[0].message.content
      history.append({"role": "assistant", "content": reply})
      return reply

  # Usage
  print(chat("What is Python?"))
  print(chat("Give me an example"))   # remembers context!
  print(chat("Now in JavaScript"))    # still remembers!

CONTEXT WINDOW WARNING:
  Every turn adds tokens. Eventually you'll hit the context limit!
  
  Strategy — Sliding Window:
    if total_tokens(history) > MAX_TOKENS * 0.8:
        # Keep system prompt + last N exchanges
        history = [history[0]] + history[-6:]

MONITORING TOKEN USAGE:
  usage = response.usage
  print(f"Prompt tokens:     {usage.prompt_tokens}")
  print(f"Completion tokens: {usage.completion_tokens}")
  print(f"Total:             {usage.total_tokens}")
```
