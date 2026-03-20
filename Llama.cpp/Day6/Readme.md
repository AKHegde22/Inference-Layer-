# Day 6: Embeddings, Structured Output, and Multimodal in llama.cpp

## Generating Embeddings

Embeddings are dense vector representations of text. Instead of generating tokens, the model outputs a float array that captures the semantic meaning of your input.

### Why embeddings?
- Semantic search — find similar documents by meaning, not keywords
- RAG (Retrieval-Augmented Generation) — retrieve relevant chunks before generating
- Clustering — group similar texts together
- Classification — use vectors as ML features

### Recommended embedding models (GGUF):

| Model | Dimensions | Notes |
|---|---|---|
| nomic-embed-text-v1.5 | 768 | Fast, excellent quality |
| mxbai-embed-large-v1 | 1024 | High quality |
| all-minilm-l6-v2 | 384 | Very fast, lightweight |
| bge-large-en-v1.5 | 1024 | Strong on benchmarks |

### Using the CLI:
```bash
./build/bin/llama-embedding \
  -m models/nomic-embed-text.Q4_K_M.gguf \
  -p "The quick brown fox"
# Outputs: float vector like [0.023, -0.14, 0.87, ...]
```

### Using llama-server API:
```bash
# Start server with embedding model
./build/bin/llama-server \
  -m models/nomic-embed.gguf --port 8080 --embeddings
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.embeddings.create(
    model="local",
    input="The quick brown fox"
)
vector = response.data[0].embedding
print(f"Dimensions: {len(vector)}")   # e.g. 768
print(f"First 5:    {vector[:5]}")
```

### Comparing embeddings with cosine similarity:
```python
import numpy as np

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Range: -1 (opposite meaning) to +1 (identical meaning)
# > 0.85 = very similar
# < 0.50 = unrelated
```

---

## Grammar-Constrained Generation (GBNF)

llama.cpp can constrain output to follow a grammar, guaranteeing structured output every time — perfect for JSON, code, or any fixed schema.

GBNF = GGML BNF (Backus-Naur Form variant). Grammar files ship with llama.cpp in the `grammars/` folder.

### Built-in grammars:
| File | Output |
|---|---|
| `grammars/json.gbnf` | Any valid JSON |
| `grammars/json_arr.gbnf` | JSON array |
| `grammars/list.gbnf` | Bullet list |
| `grammars/chess.gbnf` | Chess moves (SAN notation) |

### Using a grammar with llama-cli:
```bash
./build/bin/llama-cli \
  -m models/mistral.gguf \
  --grammar-file grammars/json.gbnf \
  -p "Return a JSON with name and age of Albert Einstein"
# Guaranteed output: {"name": "Albert Einstein", "age": 76}
```

### Writing your own GBNF:
```
# person.gbnf
root   ::= "{" ws "\"name\":" ws string "," ws "\"age\":" ws number "}"
string ::= "\"" [a-zA-Z ]+ "\""
number ::= [0-9]+
ws     ::= [ \t\n]*
```

```bash
./build/bin/llama-cli \
  -m models/mistral.gguf \
  --grammar-file person.gbnf \
  -p "Person: Marie Curie, born 1867"
```

### Using grammar via API:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "List 3 planets as JSON"}],
    "grammar": "root ::= \"[\" item+ \"]\""
  }'
```

---

## JSON Schema and Structured Output

Instead of writing raw GBNF, pass a JSON Schema and llama.cpp converts it to a grammar internally.

```python
import json
import requests

schema = {
    "type": "object",
    "properties": {
        "name":       {"type": "string"},
        "birth_year": {"type": "integer"},
        "known_for":  {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "birth_year", "known_for"]
}

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "local",
        "messages": [{"role": "user", "content": "Tell me about Nikola Tesla"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}
    }
)

data = json.loads(response.json()["choices"][0]["message"]["content"])
print(data["name"])        # "Nikola Tesla"
print(data["birth_year"])  # 1856
```

### Why this matters:

| Without grammar | With grammar |
|---|---|
| Model might return JSON... or might add prose | Always returns valid JSON — 100% guaranteed |

This is critical for any production pipeline where you parse the model's output programmatically.

---

## Function Calling (Tool Use)

Models fine-tuned for tool use can output structured function call requests. llama-server supports this for compatible models (Llama 3.1+, Mistral, Qwen2.5).

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

msg = response.choices[0].message
if msg.tool_calls:
    call = msg.tool_calls[0]
    print(call.function.name)       # "get_weather"
    print(call.function.arguments)  # '{"city": "Paris"}'
```

---

## Multimodal (Vision) Models

llama.cpp supports vision models via a separate vision projector file (`--mmproj`).

Supported models: LLaVA, BakLLaVA, MobileVLM, Qwen2-VL, InternVL, SmolVLM

### CLI usage:
```bash
./build/bin/llama-cli \
  -m models/llava-v1.6-mistral.Q4_K_M.gguf \
  --mmproj models/llava-v1.6-mistral-mmproj.gguf \
  --image /path/to/photo.jpg \
  -p "Describe this image in detail"
```

> Two files are always required: the LLM itself and the vision projector (`--mmproj`).

### Via API (base64 image):
```python
import base64

with open("photo.jpg", "rb") as f:
    base64_str = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="local",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}}
        ]
    }]
)
```
