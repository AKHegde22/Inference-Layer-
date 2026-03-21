#!/usr/bin/env bash
# Day 9 — Start the vLLM OpenAI-compatible server
#
# vLLM exposes the same API as OpenAI, so any OpenAI SDK client works out of the box.
# The server handles continuous batching and PagedAttention automatically.

# ── Basic start (Mistral 7B) ──────────────────────────────────────────────────
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000

# ── With GPU memory tuning ────────────────────────────────────────────────────
# python -m vllm.entrypoints.openai.api_server \
#   --model mistralai/Mistral-7B-Instruct-v0.2 \
#   --port 8000 \
#   --gpu-memory-utilization 0.90 \
#   --max-model-len 4096

# ── Multi-GPU (tensor parallelism across 2 GPUs) ──────────────────────────────
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Meta-Llama-3-8B-Instruct \
#   --port 8000 \
#   --tensor-parallel-size 2 \
#   --gpu-memory-utilization 0.90

# ── Quantized model (AWQ) ─────────────────────────────────────────────────────
# python -m vllm.entrypoints.openai.api_server \
#   --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
#   --port 8000 \
#   --quantization awq

# ── Gated model (Llama 3) — set HF_TOKEN first ───────────────────────────────
# export HF_TOKEN="hf_your_token_here"
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Meta-Llama-3-8B-Instruct \
#   --port 8000

# ── Key flags reference ───────────────────────────────────────────────────────
# --model                   HuggingFace model ID or local path
# --port                    Port to listen on (default: 8000)
# --host                    Host to bind (default: 0.0.0.0)
# --gpu-memory-utilization  Fraction of VRAM for KV cache (default: 0.90)
# --max-model-len           Override max context length
# --tensor-parallel-size    Number of GPUs to split model across
# --quantization            awq | gptq | fp8 | None
# --dtype                   auto | float16 | bfloat16
# --max-num-seqs            Max concurrent sequences (default: 256)
# --served-model-name       Alias shown in /v1/models response
