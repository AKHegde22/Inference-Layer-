LoRA Serving — One Base, Many Adapters
WHAT IS LoRA?
══════════════════════════════════════════════════════

LoRA (Low-Rank Adaptation) lets you fine-tune a model cheaply
by adding small trainable matrices to frozen base model weights.

MATH INTUITION:
  A weight matrix W has shape [d_out, d_in]
  Fine-tuning W directly = d_out × d_in parameters to train
  
  LoRA instead trains two small matrices:
    A: [r, d_in]   (r << d_out, d_in — the "rank")
    B: [d_out, r]
  
  Effective weight: W' = W + B × A × scale
  
  Example: W is [4096, 4096] = 16.7M params
  With rank=16: A=[16,4096] + B=[4096,16] = 131K params  ← 127× fewer!

WHY LoRA MATTERS FOR SERVING:
  One 7B base model + many LoRA adapters:
  
    Base model (7B):       4 GB  (shared in VRAM once)
    LoRA adapter (r=16):  ~50 MB (tiny, swap per-request)
  
  Use cases:
  • Multi-tenant SaaS: each customer has their own fine-tuned adapter
  • Task routing: coding adapter, legal adapter, medical adapter
  • A/B testing fine-tunes without loading multiple full models

LORA IN vLLM:

  # Start server with LoRA enabled
  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules \
      sql-lora=./adapters/sql-adapter \
      code-lora=./adapters/code-adapter \
      legal-lora=./adapters/legal-adapter \
    --max-num-seqs 256

  # Request with specific LoRA adapter
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

  # Use base model
  resp = client.chat.completions.create(
      model="meta-llama/Meta-Llama-3-8B-Instruct",
      messages=[{"role":"user","content":"Write a SQL query"}]
  )

  # Use SQL LoRA adapter — just change the model name!
  resp = client.chat.completions.create(
      model="sql-lora",   # ← adapter name as model
      messages=[{"role":"user","content":"Write a SQL query"}]
  )

KEY FLAGS:
  --enable-lora               enable LoRA support
  --max-lora-rank N           max rank of any adapter (default: 16)
  --lora-modules name=path    register adapters (can add many)
  --max-loras N               max adapters in VRAM at once (default: 1)
  --lora-dtype                adapter weight dtype (default: auto)

  Dynamic LoRA Loading
LOADING ADAPTERS AT RUNTIME (without restart)
══════════════════════════════════════════════════════

vLLM supports loading LoRA adapters dynamically via API.
No server restart needed to add or remove adapters!

LOAD ADAPTER VIA API:
  import requests

  # Load a new adapter dynamically
  requests.post("http://localhost:8000/v1/load_lora_adapter", json={
      "lora_name": "finance-lora",
      "lora_path": "/adapters/finance-adapter"
  })

  # Unload an adapter to free memory
  requests.post("http://localhost:8000/v1/unload_lora_adapter", json={
      "lora_name": "finance-lora"
  })

  # List currently loaded adapters
  resp = requests.get("http://localhost:8000/v1/models")
  for model in resp.json()["data"]:
      print(model["id"])   # shows base + all loaded adapters

HOW vLLM MANAGES LoRA IN MEMORY:

  VRAM layout with --max-loras 3:
  ┌──────────────────────────────────────────────┐
  │  Base Model Weights (frozen, always in VRAM) │
  ├──────────────────────────────────────────────┤
  │  LoRA Slot 1: sql-lora   (A+B matrices)      │
  │  LoRA Slot 2: code-lora  (A+B matrices)      │
  │  LoRA Slot 3: [empty]                         │
  └──────────────────────────────────────────────┘
  
  When slot is full and new adapter needed:
  → LRU (Least Recently Used) eviction from VRAM
  → Evicted adapter moved to CPU RAM
  → Reloaded from CPU when needed again

BATCHING REQUESTS ACROSS ADAPTERS:
  vLLM can batch requests using DIFFERENT adapters simultaneously!
  
  Iteration N:
    [Request A: sql-lora  tok5]
    [Request B: code-lora tok3]   ← different adapters, same batch!
    [Request C: base-model tok8]
  
  How? The adapter weights are applied per-sample in the batch
  using a technique called "punica" kernels (CUDA batch LoRA).
  
  This means you get full throughput even with many adapter users.

TRAINING YOUR OWN LoRA:
  pip install peft transformers

  from peft import LoraConfig, get_peft_model
  from transformers import AutoModelForCausalLM

  config = LoraConfig(
      r=16,                    # rank
      lora_alpha=32,           # scaling factor
      target_modules=["q_proj","v_proj"],  # which layers
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM"
  )
  model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
  model = get_peft_model(model, config)
  model.print_trainable_parameters()
  # trainable params: 6,815,744 || all params: 8,036,335,616 || 0.08%

  Speculative Decoding
THE PROBLEM: TOKEN GENERATION IS SEQUENTIAL
══════════════════════════════════════════════════════

Normal autoregressive generation:
  Step 1: Run full 8B model → generate token 1
  Step 2: Run full 8B model → generate token 2
  Step 3: Run full 8B model → generate token 3
  ...
  
  GPU is bottlenecked by memory bandwidth, not compute.
  Each step loads ALL 8B weights from VRAM just to generate 1 token.
  This is called "memory-bandwidth bound" inference.

SPECULATIVE DECODING SOLUTION:
  Use a tiny "draft" model to GUESS multiple tokens ahead.
  Use the large "target" model to VERIFY all guesses in one pass.

  Step-by-step:
  1. Draft model (e.g. 100M params) generates K token guesses fast
     → tok1_guess, tok2_guess, tok3_guess, tok4_guess  (K=4)
  
  2. Target model (8B) verifies ALL 4 guesses in ONE forward pass
     (parallel verification is possible because we have the guesses)
  
  3. Accept guesses up to first mismatch:
     If tok1,tok2 correct but tok3 wrong:
     → Accept tok1, tok2, reject from tok3 onward
     → Net result: 2 tokens generated in 1 big-model pass!
  
  4. Repeat from accepted position

SPEEDUP INTUITION:
  Normal:  1 big-model pass = 1 token
  Spec:    1 big-model pass = K×acceptance_rate tokens
  
  With K=4, acceptance_rate=0.8:
    Expected tokens per pass = 4 × 0.8 = 3.2 tokens
    Speedup ≈ 3.2× vs 1× normal
  
  Best case: similar topics (e.g. code), high acceptance rate
  Worst case: creative/random outputs, low acceptance rate

USING SPECULATIVE DECODING IN vLLM:

  # Method 1: Separate draft model
  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-model meta-llama/Meta-Llama-3.2-1B \
    --num-speculative-tokens 5

  # Method 2: NGram draft (no extra model needed!)
  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-model "[ngram]" \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 4

  # Method 3: MedusaHead (trained draft heads on same model)
  vllm serve FasterDecoding/medusa-1.0-llama-2-7b-chat \
    --speculative-model medusa

  # Python
  llm = LLM(
      model="meta-llama/Meta-Llama-3-8B-Instruct",
      speculative_model="meta-llama/Meta-Llama-3.2-1B",
      num_speculative_tokens=5,
  )

WHEN TO USE SPECULATIVE DECODING:
  ✓  Low-latency chatbots (reduces TTFT and TPOT)
  ✓  Code generation (repetitive patterns, high acceptance)
  ✓  Constrained/templated outputs
  ✗  High-throughput batch processing (overhead not worth it)
  ✗  Very creative/random outputs (low acceptance rate)

Prefix Caching — Reuse KV Computation
THE INSIGHT: SAME PREFIX = SAME KV CACHE
══════════════════════════════════════════════════════

Every time you send a request, the model processes all tokens:
  [system prompt] + [conversation history] + [new user message]
  
  For a RAG pipeline with 2000-token context:
  Without prefix caching: compute KV for ALL 2000 tokens every request
  With prefix caching:    reuse KV for the 2000 shared tokens!

HOW IT WORKS:

  vLLM hashes the content of each KV cache block.
  If the hash matches a previously computed block, reuse it.

  Request 1: [System: You are a SQL expert (500 tok)] [User: Q1]
  → Compute KV for all 500 + Q1 tokens. Store hash.
  
  Request 2: [System: You are a SQL expert (500 tok)] [User: Q2]
  → Hash matches for the 500-token prefix!
  → SKIP computation for first 500 tokens
  → Only compute KV for Q2 tokens
  → TTFT drops dramatically!

ENABLING PREFIX CACHING:

  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --enable-prefix-caching

  # Python
  llm = LLM(
      model="meta-llama/Meta-Llama-3-8B-Instruct",
      enable_prefix_caching=True,
  )

BEST USE CASES:

  1. RAG pipelines
     [long retrieved context] + [different questions]
     → Huge win: context reused across all queries

  2. Multi-turn chat
     [growing history] + [new message]
     → Reuse all prior turns' KV cache

  3. Multi-tenant with shared system prompts
     [company system prompt] + [user message]
     → System prompt computed once, shared across users

  4. Few-shot prompting
     [long few-shot examples] + [new input]
     → Examples computed once per batch

MEASURING PREFIX CACHE HIT RATE:
  curl http://localhost:8000/metrics | grep prefix_cache
  # vllm:gpu_prefix_cache_hit_rate 0.847  ← 84.7% cache hits!

  # In Python response object
  usage = response.usage
  # Check prompt_tokens vs cached_tokens if API exposes it

CHUNKED PREFILL — RELATED OPTIMIZATION:
  --enable-chunked-prefill
  
  Splits long prompts into chunks processed across iterations.
  Prevents long prompts from blocking the GPU for many steps.
  Especially useful when mixing short and long context requests.

Combining Advanced Features
PRODUCTION RECIPE — ALL FEATURES TOGETHER
══════════════════════════════════════════════════════

Here is a production-grade vLLM command combining everything
from Days 8–11:

  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    # Quantization (Day 10)
    --quantization awq \
    --dtype auto \
    \
    # Memory (Days 8-9)
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    \
    # Prefix Caching (Day 11)
    --enable-prefix-caching \
    --enable-chunked-prefill \
    \
    # LoRA (Day 11)
    --enable-lora \
    --max-lora-rank 32 \
    --max-loras 4 \
    --lora-modules \
      sql=./adapters/sql \
      code=./adapters/code \
    \
    # Speculative Decoding (Day 11)
    --speculative-model "[ngram]" \
    --num-speculative-tokens 5 \
    \
    # Server (Day 9)
    --host 0.0.0.0 --port 8000 \
    --api-key secret-key \
    --served-model-name llama3-prod

FEATURE INTERACTION NOTES:
  ┌───────────────────────────┬──────────────────────────────┐
  │ Feature combo             │ Notes                        │
  ├───────────────────────────┼──────────────────────────────┤
  │ AWQ + prefix caching      │ Works perfectly ✓            │
  │ LoRA + prefix caching     │ Works, cache per adapter ✓   │
  │ Spec decoding + LoRA      │ Experimental in some builds  │
  │ Spec decoding + prefix    │ Compatible ✓                 │
  │ Chunked prefill + spec    │ May conflict — test first    │
  └───────────────────────────┴──────────────────────────────┘

MONITORING ALL FEATURES:
  import requests

  metrics_text = requests.get("http://localhost:8000/metrics").text

  key_metrics = [
      "vllm:gpu_cache_usage_perc",       # KV cache fullness
      "vllm:gpu_prefix_cache_hit_rate",  # prefix cache effectiveness
      "vllm:num_requests_running",       # active requests
      "vllm:spec_decode_draft_acceptance_rate",  # spec decode quality
      "vllm:e2e_request_latency_seconds", # end-to-end latency
  ]

DECIDING WHICH FEATURES TO ENABLE:
  Has shared system prompt / RAG?    → prefix caching (always)
  Multiple fine-tuned tasks?         → LoRA serving
  Latency-sensitive chatbot?         → speculative decoding
  Long documents?                    → chunked prefill
  Limited VRAM?                      → AWQ/GPTQ quantization
  All of the above?                  → combine carefully, test!
