# Day 2: Quantization and Model Conversion with llama.cpp

## Why Quantization?

A 7B parameter model in full precision (**FP32**) takes:
`7,000,000,000 × 4 bytes = ~28 GB of RAM`

That's too big for most consumer hardware. Quantization shrinks it significantly by storing a lower-precision approximation instead of the exact float value of each weight. This trades a small amount of accuracy for a massive reduction in memory and a big speedup on CPUs.

### Precision vs. Memory (Example for 7B Model):
*   **FP32**  → `4 bytes` per weight  → **28 GB** *(full quality)*
*   **FP16**  → `2 bytes` per weight  → **14 GB** *(nearly same quality)*
*   **INT8**  → `1 byte` per weight   → **7 GB** *(slight quality drop)*
*   **INT4**  → `0.5 bytes` per weight → **3.5 GB** *(noticeable but usable)*
*   **INT2**  → `0.25 bytes` per weight → **1.8 GB** *(significant quality loss)*

---

## llama.cpp Quantization Types

`llama.cpp` uses a specific naming scheme for its quantized models: `Q{bits}_{variant}`.

### Common Types on HuggingFace:
*   **Q4_0** — 4-bit, simple, oldest method, fastest but lower quality
*   **Q4_1** — 4-bit, adds a bias term, slightly better than Q4_0
*   **Q4_K_S** — 4-bit K-quant, Small (~3.8 GB)
*   **Q4_K_M** — 4-bit K-quant, Medium (~4.1 GB) ← *Most popular*
*   **Q4_K_L** — 4-bit K-quant, Large
*   **Q5_K_M** — 5-bit K-quant, Medium (~4.8 GB)
*   **Q6_K** — 6-bit K-quant (~5.5 GB)
*   **Q8_0** — 8-bit, near-lossless (~7.2 GB)

### What are K-quants?
**K-quants** ("k" = k-means clustering) group weights into blocks and find optimal quantization points per block using k-means. This is much smarter than naive rounding, resulting in better quality at the same bit-width.

### What do S / M / L mean?
Some layers in a neural network are more sensitive to quantization than others. K-quant variants use higher precision on sensitive layers:
*   **_S (Small):** All layers use the base bit-width.
*   **_M (Medium):** Attention layers get slightly higher precision.
*   **_L (Large):** More layers get higher precision.

> 💡 **Recommendation for most use cases:** `Q4_K_M` or `Q5_K_M`

---

## Converting HuggingFace → GGUF

If a model doesn't have a GGUF version available, you can easily create one yourself.

### Step 1: Install Python dependencies
```bash
pip install -r requirements.txt
# Key packages: transformers, sentencepiece, numpy, torch
```

### Step 2: Download the HuggingFace model
```python
from huggingface_hub import snapshot_download

snapshot_download(
    "mistralai/Mistral-7B-Instruct-v0.2",
    local_dir="./hf-model"
)
```

### Step 3: Convert to FP16 GGUF
```bash
python convert_hf_to_gguf.py ./hf-model \
  --outtype f16 \
  --outfile ./models/mistral-f16.gguf
```

### Step 4: Quantize to your desired format
```bash
./build/bin/llama-quantize \
  ./models/mistral-f16.gguf \
  ./models/mistral-q4_k_m.gguf \
  Q4_K_M
```

### Step 5: Verify it works
```bash
./build/bin/llama-cli -m ./models/mistral-q4_k_m.gguf \
  -p "Hello!" -n 50
```

---

## Comparing Quantization Quality

How do we measure the quality loss resulting from quantization?

### Method 1: Perplexity (Automatic Metric)
Lower perplexity = better model quality.

```bash
./build/bin/llama-perplexity \
  -m models/mistral-q4_k_m.gguf \
  -f wikitext-2-raw/wiki.test.raw
```

**Typical perplexity scores for Mistral 7B:**
*   **F16** → `~5.1` *(Reference)*
*   **Q8_0** → `~5.1` *(Virtually identical)*
*   **Q6_K** → `~5.1` *(Excellent)*
*   **Q5_K_M** → `~5.2` *(Very good)*
*   **Q4_K_M** → `~5.3` *(Good, recommended)*
*   **Q3_K_M** → `~5.7` *(Acceptable)*
*   **Q2_K** → `~7.0` *(Noticeable degradation)*

### Method 2: Side-by-side Prompt Testing
Run the same prompt on different quantizations (e.g., Q4 vs. Q8 vs. F16) and compare the outputs manually. 
**Tip:** Be sure to test complex reasoning tasks, writing code, and solving math problems.

### Rule of Thumb:
*   **For everyday chat:** `Q4_K_M` is fine.
*   **For coding / reasoning:** `Q5_K_M` or `Q6_K`.
*   **For benchmarking:** `Q8_0` or `F16`.
*   **For RAM-constrained systems:** `Q3_K_M` (last resort).
