# Day 5: GPU Acceleration and Performance Tuning in llama.cpp

## GPU Backends

llama.cpp supports multiple GPU backends, chosen at build time.

| Backend | Platform | Build Flag |
|---|---|---|
| CUDA | NVIDIA GPUs | `-DGGML_CUDA=ON` |
| Metal | Apple Silicon | enabled by default on macOS |
| ROCm/HIP | AMD GPUs | `-DGGML_HIP=ON` |
| Vulkan | Cross-platform | `-DGGML_VULKAN=ON` |
| SYCL | Intel GPUs | `-DGGML_SYCL=ON` |
| CPU (BLAS) | All | `-DGGML_BLAS=ON` |

### Build commands:
```bash
# CUDA (NVIDIA)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Metal (macOS — auto-detected, no flag needed)
cmake -B build
cmake --build build --config Release -j$(nproc)

# ROCm (AMD)
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1100"
cmake --build build --config Release -j$(nproc)

# Vulkan (cross-platform fallback)
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
```

### Verify GPU is detected:
```bash
./build/bin/llama-cli -m model.gguf -ngl 1 -p "hi" -n 5
# Look for: "llm_load_tensors: offloading N layers to GPU"
# If you see: "ggml_cuda_init: CUDA not found" → rebuild with the CUDA flag
```

---

## GPU Layer Offloading (`-ngl`)

The `-ngl` flag (n-gpu-layers) controls how many transformer layers are offloaded to the GPU.

A 7B model has 32 transformer layers plus embeddings and an output head. Each layer's weights live in either RAM or VRAM.

```
-ngl 0   → all layers on CPU (pure CPU inference)
-ngl 16  → first 16 layers on GPU, rest on CPU
-ngl 32  → all transformer layers on GPU
-ngl 99  → everything on GPU including embeddings ← use this
```

### VRAM requirements for full offload:

| Model | Q4_K_M Size | Min VRAM |
|---|---|---|
| Llama 3.2 3B | ~2.0 GB | 3 GB |
| Mistral / Llama 7B | ~4.1 GB | 6 GB |
| Llama 3 8B | ~4.9 GB | 6 GB |
| Llama 3 13B | ~7.9 GB | 10 GB |
| Llama 3 70B | ~40 GB | 48 GB (or 2×24GB) |

### Partial offload strategy (limited VRAM):

If your GPU can't fit the whole model, offload as many layers as possible and let the rest run on CPU.

```bash
-ngl 20   # experiment to find the sweet spot
```

How to find the optimal value:
1. Start with `-ngl 99` and check if it crashes (OOM)
2. If OOM, reduce by 4–8 layers at a time
3. Monitor VRAM with `nvidia-smi` (NVIDIA) or `sudo powermetrics` (Mac)

### Performance impact:

| Setup | Speed |
|---|---|
| CPU only | ~5–15 tokens/sec (8-core CPU) |
| Partial GPU (16/32 layers) | ~20–35 tokens/sec |
| Full GPU offload | ~40–120 tokens/sec |

---

## Performance Tuning Flags

### `--flash-attn` / `-fa`
Implements the FlashAttention algorithm.
- 20–40% less VRAM for attention computation
- 10–30% faster on long contexts
- Enables much longer context windows

```bash
./build/bin/llama-server -m model.gguf -ngl 99 --flash-attn
```

### `--no-mmap` (default: mmap ON)
`mmap` = memory-mapped file I/O. By default the model file is mapped rather than fully loaded — faster startup but slower first inference due to page faults.

```bash
--no-mmap   # load entire model into RAM first
            # slower startup, but faster first token
```

### `--mlock`
Locks model memory so the OS can't swap it to disk. Prevents latency spikes. May require `sudo` and sufficient RAM.

### `--numa distribute`
For multi-socket CPU servers. Distributes model weights across NUMA nodes.

### `-t` / `-tb` (threads)
```
-t  N   → threads for token generation
-tb N   → threads for prompt processing (batch)
```

> Set `-t` to your physical core count, not logical/hyperthreaded. Example: 8-core CPU with HT → use `-t 8`, not `-t 16`.

### Production command (fully tuned):
```bash
./build/bin/llama-server \
  -m models/mistral.Q4_K_M.gguf \
  -ngl 99 \
  --flash-attn \
  --mlock \
  -c 8192 \
  -t 8 \
  --parallel 4 \
  --cont-batching \
  --host 0.0.0.0 --port 8080
```

---

## Benchmarking and Profiling

llama.cpp ships with a dedicated benchmarking tool: `llama-bench`.

```bash
# Basic benchmark (prompt eval + generation)
./build/bin/llama-bench -m model.gguf

# Test specific context and batch sizes
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -b 512 -ngl 99

# Compare CPU vs GPU offload levels
./build/bin/llama-bench -m model.gguf -ngl 0,16,32,99
```

Output columns:
- `pp` — prompt processing speed (tokens/sec) — batch speed
- `tg` — token generation speed (tokens/sec) — chat speed

### Key metrics:

| Metric | What it measures |
|---|---|
| TTFT (Time To First Token) | How long until you see any output — dominated by `pp` speed |
| TPOT (Time Per Output Token) | Gap between each generated token — dominated by `tg` speed |
| Throughput | Tokens/sec across concurrent users — most important for API servers |

### Monitoring GPU:
```bash
# NVIDIA — live stats
nvidia-smi dmon -s u

# NVIDIA — one-shot
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv

# macOS Apple Silicon
sudo powermetrics --samplers gpu_power -i 1000

# llama-server metrics endpoint
curl http://localhost:8080/metrics
```
