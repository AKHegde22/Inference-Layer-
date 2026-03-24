# Day 10: Quantization in vLLM — AWQ, GPTQ, FP8, BitsAndBytes

You already know quantization from Week 1 (GGUF Q4/Q5/Q8 in llama.cpp). vLLM uses completely different formats — GPU-native ones built for CUDA kernels, not CPU arithmetic. Same goal, different execution.

---

## llama.cpp vs vLLM Quantization

```
  llama.cpp (Week 1)              vLLM (Week 2)
  ──────────────────────────────  ──────────────────────────────
  GGUF format (Q4_K_M etc.)       HuggingFace format (.safetensors)
  CPU-optimized kernels            GPU-optimized CUDA kernels
  Integer arithmetic (INT4)        INT4 weights + FP16 activations
  Single-file model                Separate weight + config files
  Great for consumer CPUs          Great for production GPU serving
```

### VRAM requirements for Mistral 7B

| Format | VRAM | Quality | Speed vs FP16 |
|---|---|---|---|
| FP32 | 28 GB | Reference | 0.5× |
| FP16 / BF16 | 14 GB | ≈ FP32 | 1× (baseline) |
| GPTQ INT4 | 4 GB | Good | 1.2× |
| AWQ INT4 | 4 GB | Better | 1.3× |
| FP8 | 7 GB | ≈ FP16 | 1.5× (H100+) |
| BitsAndBytes | 5 GB | Good | 0.9× |

GPU quantization stores weights as INT4 but upcasts to FP16 before the matrix multiply. GPUs are optimized for FP16 math, not INT4 — so this is faster than running INT4 arithmetic directly.

---

## File Structure

```
Day10/
├── quant-format-selct.py    ← Recommends the right format given GPU + model size
├── vram-Budget.py           ← Calculates exact VRAM breakdown (weights + KV cache + overhead)
├── quality-benchmarks.py    ← Compares output consistency across quantization formats
└── quant-confg-inspector.py ← Parses HuggingFace config.json to detect quantization
```

---

## Setup

```bash
pip install vllm openai

# For quantizing your own models
pip install autoawq        # AWQ quantization
pip install auto-gptq      # GPTQ quantization
pip install llmcompressor  # FP8 static quantization
```

---

## quant-format-selct.py — Format Recommender

Given your GPU's VRAM, model size, GPU architecture, and priority, returns the best quantization format with reasoning.

```bash
python quant-format-selct.py
```

```python
select_quantization(gpu_vram_gb=80, model_params_b=70, gpu_arch="h100",   priority="speed")
select_quantization(gpu_vram_gb=24, model_params_b=7,  gpu_arch="rtx4090",priority="quality")
select_quantization(gpu_vram_gb=8,  model_params_b=7,  gpu_arch="a10g",   priority="memory")
select_quantization(gpu_vram_gb=6,  model_params_b=13, gpu_arch="rtx3090",priority="quality")
select_quantization(gpu_vram_gb=4,  model_params_b=70, gpu_arch="other",  priority="memory")
```

### Expected output

```
{'format': 'fp8',          'vram_needed': 70.0, 'reason': 'H100 native FP8 — best quality and fastest on this GPU.'}
{'format': 'fp16',         'vram_needed': 14.0, 'reason': 'Full precision FP16 fits — no quality loss.'}
{'format': 'awq',          'vram_needed': 4.2,  'reason': 'AWQ INT4 — best quality INT4, fast GEMM kernels.'}
{'format': 'gptq',         'vram_needed': 7.8,  'reason': 'GPTQ INT4 — good quality, widely compatible.'}
{'format': 'does_not_fit', 'vram_needed': 45.5, 'reason': 'Model needs ~45.5GB minimum but only 4GB available.'}
```

### Decision logic

```
Have H100/H200?
├── YES → FP8
└── NO
    ├── FP16 fits?  → FP16 (no quality loss)
    ├── AWQ fits?   → AWQ (best INT4 quality)
    ├── GPTQ fits?  → GPTQ (good, widely supported)
    ├── BnB fits?   → BitsAndBytes (no pre-quant needed)
    └── Nothing fits → does_not_fit
```

---

## vram-Budget.py — VRAM Budget Planner

Breaks down exactly where your VRAM goes: model weights, KV cache, and CUDA overhead.

```bash
python vram-Budget.py
```

```python
planner = VRAMBudgetPlanner(total_vram_gb=24)
planner.print_plan(
    params_b=7, quant_format="awq",
    ctx_len=8192, n_layers=32, n_heads=8, head_dim=128,
    max_seqs=32
)
```

### Expected output

```
════════════════════════════════════════════════
  VRAM Budget Plan  [FITS]
  7B params | AWQ | ctx=8192 | seqs=32
════════════════════════════════════════════════
  Model Weights              4.20 GB
  KV Cache (32 seqs)         8.59 GB
  CUDA Overhead              1.50 GB
  ──────────────────────────────────
  Total Required            14.29 GB
  Available                 24.00 GB
  Headroom                  +9.71 GB
════════════════════════════════════════════════
```

### How each component is calculated

| Component | Formula |
|---|---|
| Model weights | `params_b × quant_multiplier` |
| KV cache | `2 × layers × heads × head_dim × ctx_len × dtype_bytes × n_seqs` |
| CUDA overhead | ~1.5 GB fixed (kernels, activations, buffers) |

The KV cache grows linearly with context length and number of concurrent sequences — it's often larger than the model weights at high concurrency.

---

## quality-benchmarks.py — Cross-Format Quality Comparison

Runs the same prompts against multiple vLLM servers (each loaded with a different quantization format) and measures output consistency relative to FP16.

```bash
python quality-benchmarks.py
```

Requires separate vLLM server instances running on different ports:

```bash
# Terminal 1 — FP16 reference
python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8001

# Terminal 2 — AWQ
python -m vllm.entrypoints.openai.api_server --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ --quantization awq --port 8002

# Terminal 3 — GPTQ
python -m vllm.entrypoints.openai.api_server --model TheBloke/Mistral-7B-Instruct-v0.2-GPTQ --quantization gptq --dtype float16 --port 8003
```

### Expected output

```
══════════════════════════════════════════════════════
  Quantization Quality Benchmark
══════════════════════════════════════════════════════
  Model                  Format          Consistency
  ────────────────────────────────────────────────────
  FP16 (Reference)       fp16                100%  ██████████
  AWQ INT4               awq                  90%  █████████
  GPTQ INT4              gptq                 80%  ████████
  BitsAndBytes           bitsandbytes         70%  ███████
══════════════════════════════════════════════════════
```

Consistency is measured by word overlap with the FP16 reference. Temperature is set to 0.0 for deterministic outputs.

---

## quant-confg-inspector.py — Config Inspector

Parses a HuggingFace `config.json` (or a dict) to detect quantization format, bit width, group size, and vLLM compatibility — without loading the model.

```bash
python quant-confg-inspector.py
```

```python
insp = QuantInspector()
insp.load_config(awq_config)   # dict or path to config.json
insp.summarize()
insp.is_compatible_with_vllm()
```

### Expected output

```
══════════════════════════════════════════════
  Model Config Summary
══════════════════════════════════════════════
  Architecture         MistralForCausalLM
  dtype                float16
  Vocab Size           32000
  Context Length       32768
  Quantized            True
  Format               awq
  Bits                 4
  Group Size           128
  Zero Point           True
══════════════════════════════════════════════
vLLM compatible: True

══════════════════════════════════════════════
  Model Config Summary
══════════════════════════════════════════════
  Architecture         LlamaForCausalLM
  dtype                bfloat16
  Vocab Size           128256
  Context Length       8192
  Quantized            False
══════════════════════════════════════════════
vLLM compatible: True
```

Supported formats checked against vLLM's known list: `awq`, `gptq`, `fp8`, `bitsandbytes`, `squeezellm`.

---

## The Four Formats

### AWQ — Activation-Aware Weight Quantization

Key insight: ~1% of weights are "salient" — they correspond to large activations and cause most of the quality loss when quantized. AWQ finds these weights via calibration data, scales them up before quantization, then scales activations down by the same factor. The output is identical but the important weights get better resolution.

```python
# Load pre-quantized AWQ model
llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    quantization="awq",
    dtype="auto",
)
```

```bash
# Quantize your own model
pip install autoawq
# See Readme theory section for full AutoAWQ script
```

### GPTQ — Post-Training Quantization

Uses second-order information (Hessian matrix) to minimize reconstruction error layer by layer. More compute-intensive to quantize than AWQ but very principled. Slightly slower inference than AWQ.

```python
llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    quantization="gptq",
    dtype="float16",   # GPTQ needs explicit float16
)
```

### FP8 — 8-bit Floating Point

Native hardware support on H100/H200/GB200. Uses floating point math (not integer) so numerical stability is better than INT4. No calibration needed for dynamic FP8. Quality is nearly identical to FP16.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    quantization="fp8",
    dtype="bfloat16",
)
```

### BitsAndBytes — Quantize at Load Time

No pre-quantized model needed. Quantizes any HuggingFace model on the fly at load time. Slower inference than AWQ/GPTQ (no custom CUDA kernels), but useful for models that don't have AWQ/GPTQ versions yet.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    dtype="float16",
)
```

---

## AWQ vs GPTQ

| Feature | AWQ | GPTQ |
|---|---|---|
| Quality | Slightly better | Good |
| Quantization speed | Faster | Slower (Hessian) |
| Inference speed | Faster (GEMM) | Slightly slower |
| vLLM support | Excellent | Excellent |
| Bit widths | 4-bit | 2, 3, 4, 8-bit |

---

## GPU Support Matrix

| Quantization | A10G | A100 | H100 | RTX 4090 |
|---|---|---|---|---|
| AWQ INT4 | ✓ | ✓ | ✓ | ✓ |
| GPTQ INT4 | ✓ | ✓ | ✓ | ✓ |
| FP8 (native) | ✗ | ✗ | ✓ | ✗ |
| FP8 (emulated) | ✓ | ✓ | ✓ | ✓ |
| BitsAndBytes | ✓ | ✓ | ✓ | ✓ |
