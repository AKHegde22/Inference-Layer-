# Day 8: Why vLLM Exists — PagedAttention & Continuous Batching

vLLM is a high-throughput inference engine built at UC Berkeley (2023). It solves two fundamental problems with naive HuggingFace serving: static batching and KV cache memory waste. This day covers the core theory and simulates the key mechanisms in pure Python.

---

## The Problem with Naive Serving

```
STATIC BATCHING (HuggingFace model.generate())
──────────────────────────────────────────────
  All sequences in a batch must finish together.
  If Request A needs 10 tokens and Request B needs 500 tokens,
  Request A must WAIT for B before new requests can start.

  t=0  [A: ████░░░░░░░░░░░░░░░░]  ← waiting idle
  t=0  [B: ████████████████████]
  t=1  [C starts only now]         ← wasted GPU cycles

KV CACHE MEMORY WASTE
──────────────────────────────────────────────
  KV cache is pre-allocated for the MAXIMUM context length.
  A request using 200 of 4096 tokens wastes 95% of that VRAM.

  Real-world HuggingFace generate():
    GPU memory utilization: ~20–40%
    Throughput: 3–5 requests/sec (A100)

vLLM achieves:
    GPU memory utilization: ~90%+
    Throughput: 20–30× higher
    Latency:    2–5× lower under load
```

---

## File Structure

```
Day8/
├── hfvsvllm.py         ← Benchmark: HuggingFace vs vLLM throughput
├── pagedattention.py   ← Annotated explanation of PagedAttention internals
├── simulatevLLM.py     ← Pure-Python simulation of the continuous batch scheduler
└── slidingParams.py    ← SamplingParams presets for different use cases
```

---

## Setup

```bash
# vLLM requires Linux or WSL2, Python 3.9–3.12, CUDA 11.8+
conda create -n vllm python=3.11 -y
conda activate vllm

pip install vllm transformers torch
```

> `hfvsvllm.py` requires a CUDA GPU. The other three files run on CPU with no GPU needed.

---

## hfvsvllm.py — Benchmark: HuggingFace vs vLLM

Runs 20 prompts through both HuggingFace `pipeline` and vLLM `LLM.generate()`, then prints a side-by-side throughput comparison.

```python
def benchmark_hf(prompts, model_name="gpt2"):
    # Uses transformers pipeline with static batching
    ...

def benchmark_vllm(prompts, model_name="gpt2"):
    # Uses vLLM with PagedAttention + continuous batching
    ...
```

### Running it

```bash
python hfvsvllm.py
```

### Expected output

```
════════════════════════════════════════════════════════
  Batch Inference Benchmark  (20 prompts)
════════════════════════════════════════════════════════
  Metric               HuggingFace           vLLM
────────────────────────────────────────────────────────
  Total Time (s)              8.412          0.631
  Throughput (req/s)           2.38          31.70
  Avg Latency (s)             0.421          0.032
────────────────────────────────────────────────────────
  vLLM Speedup                                13.3×
════════════════════════════════════════════════════════
```

The speedup varies by GPU and model size. On larger models (7B+) the gap widens significantly.

---

## pagedattention.py — PagedAttention Internals

Three annotated functions that explain the core concepts. No execution needed — the docstrings are the content.

### KV Cache Blocks

```
block_size = 16  →  each block holds K and V vectors for 16 tokens

VRAM block pool:  [B0][B1][B2][B3][B4][B5][B6][B7][B8][B9]...

Request A (50 tokens → 4 blocks):
  logical[0] → B2
  logical[1] → B5
  logical[2] → B0   ← blocks are scattered, not contiguous
  logical[3] → B8

Request B (20 tokens → 2 blocks):
  logical[0] → B1
  logical[1] → B7
```

### Why block_size=16 is the default

| block_size | Flexibility | Overhead | GPU efficiency |
|---|---|---|---|
| 1 | Maximum | Very high | Poor |
| 16 | Good | Low | Good (vLLM default) |
| 32 | Less | Minimal | Better |

### Fragmentation comparison

| Approach | Memory waste |
|---|---|
| Naive (pre-allocate max_len) | ~70% wasted |
| PagedAttention (on-demand blocks) | ~4% wasted (at most block_size-1 tokens) |

```bash
python pagedattention.py
# No output — read the docstrings directly
```

---

## simulatevLLM.py — Continuous Batch Scheduler

A pure-Python simulation of vLLM's iteration-level scheduler. No GPU required.

```python
class ContinuousBatchScheduler:
    def __init__(self, max_running=4, max_blocks=20, block_size=16):
        ...

    def add_request(self, req_id, prompt_len, max_new_tokens):
        # Adds a request to the waiting queue
        ...

    def step(self):
        # One scheduler iteration:
        # 1. Finish completed requests, free their blocks
        # 2. Promote waiting → running while block budget allows
        # 3. Increment token count for all running requests
        ...
```

### How continuous batching works

```
Step 1:  running=[R1, R2, R3]  waiting=[R4, R5, R6]
Step 2:  R2 finishes → R4 promoted immediately
Step 3:  running=[R1, R3, R4]  ← no idle cycles between batches
```

This is the key difference from static batching: new requests slot in the moment a slot opens, every single iteration.

### Running it

```bash
python simulatevLLM.py
```

### Expected output

```
Step 1: scheduled=['R1', 'R2', 'R3'] finished=[]
  Running (3): ['R1', 'R2', 'R3']
  Waiting (3): ['R4', 'R5', 'R6']
  Blocks used: 6/15

Step 2: scheduled=[] finished=[]
  Running (3): ['R1', 'R2', 'R3']
  Waiting (3): ['R4', 'R5', 'R6']
  Blocks used: 6/15

Step 3: scheduled=[] finished=['R2']
  Running (2): ['R1', 'R3']
  Waiting (3): ['R4', 'R5', 'R6']
  Blocks used: 4/15

Step 4: scheduled=['R4'] finished=[]
  Running (3): ['R1', 'R3', 'R4']
  Waiting (2): ['R5', 'R6']
  Blocks used: 8/15
...
```

The scheduler tracks three queues — `waiting`, `running`, `finished` — and a block budget. When a request finishes, its blocks are freed and the next waiting request can be promoted.

---

## slidingParams.py — SamplingParams Presets

Defines five ready-to-use `SamplingParams` configurations and a `describe_params()` helper that explains each parameter.

### Presets

| Use case | temperature | top_k | max_tokens | Notes |
|---|---|---|---|---|
| `chat` | 0.7 | 40 | 512 | Balanced, conversational |
| `code` | 0.1 | 10 | 1024 | Near-deterministic, longer output |
| `creative` | 1.0 | 0 | 800 | High randomness, no top-k limit |
| `factual` | 0.0 | 1 | 256 | Greedy decoding, shortest output |
| `batch_eval` | 0.0 | 1 | 64 | Reproducible evals with fixed seed |

### Usage

```python
from slidingParams import build_sampling_params, describe_params

# Get a preset
params = build_sampling_params("code")

# Override specific values
params = build_sampling_params("chat", temperature=0.3, max_tokens=100)

# Print a human-readable explanation of each parameter
describe_params(params)
```

### Running it

```bash
python slidingParams.py
```

### Expected output

```python
{'temperature': 0.1, 'top_p': 0.95, 'top_k': 10, 'max_tokens': 1024,
 'repetition_penalty': 1.0, 'stop': ['<|endoftext|>'], 'use_case': 'code'}

{'temperature': 0.3, 'top_p': 0.9, 'top_k': 40, 'max_tokens': 100,
 'repetition_penalty': 1.1, 'use_case': 'chat'}

SamplingParams for use_case='code':
──────────────────────────────────────────────────
  temperature            = 0.1        # Controls randomness. Low=deterministic, high=creative.
  top_p                  = 0.95       # Nucleus sampling threshold — cumulative prob cutoff.
  top_k                  = 10         # Limit candidates to top K tokens (0=disabled).
  max_tokens             = 1024       # Maximum new tokens to generate.
  repetition_penalty     = 1.0        # Penalizes repeated tokens. >1.0 reduces repetitive loops.
  stop                   = ['<|endoftext|>']  # Stop generation when any of these strings appear.
```

---

## Core Concepts Summary

### PagedAttention

Applies OS virtual memory paging to the KV cache. Physical VRAM is divided into fixed-size blocks (default 16 tokens). Each request gets a block table mapping its logical token positions to physical blocks — blocks don't need to be contiguous.

```
Traditional:  [Request A — 4096 slots reserved — 200 used — 3896 WASTED]
PagedAttention: [B1][B2]...[B13]  ← exactly 13 blocks for 200 tokens, nothing more
```

### Continuous Batching

Every forward pass, the scheduler can add newly arrived requests and remove finished ones. The GPU never idles waiting for a slow request to finish.

```
Static:     [A████████████████][B████    ]  ← B waits for A
Continuous: [A tok1][B tok1][C tok1]
            [A tok2][B tok2][C tok2]  ← B finishes
            [A tok3][D tok1][C tok3]  ← D inserted immediately
```

### vLLM Scheduler Queues

| Queue | Description |
|---|---|
| `waiting` | Requests not yet started |
| `running` | Requests currently being processed |
| `swapped` | Requests preempted to CPU RAM (VRAM full) |

---

## Using vLLM for Real Inference

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    gpu_memory_utilization=0.90,
    max_model_len=4096,
)

params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)

outputs = llm.generate(
    ["What is PagedAttention?", "Explain transformers simply"],
    params
)

for out in outputs:
    print(out.prompt[:40], "→", out.outputs[0].text[:80])
```

For gated models (Llama 3, etc.):

```bash
export HF_TOKEN="hf_your_token_here"
python your_script.py
```
