```markdown
# Day 1: Introduction to llama.cpp and GGUF

## What is llama.cpp?

`llama.cpp` is a high-performance C/C++ library designed to run Large Language Models (LLMs) locally on consumer-grade hardware. It allows you to run models on your own CPU or GPU without requiring cloud APIs or enterprise-grade data centers.

Originally created by **Georgi Gerganov** in March 2023 to run Meta's LLaMA model on a MacBook, it has since evolved into a cornerstone of the local LLM ecosystem.

### Key Benefits:
*   **Privacy:** Your prompts and data never leave your local machine.
*   **Cost:** Eliminate expensive API subscription fees.
*   **Offline Access:** Works entirely without an internet connection.
*   **Performance:** Highly optimized C++ utilizing SIMD, BLAS, and hardware acceleration (Metal, CUDA, Vulkan, ROCm).

---

## Understanding GGUF (GGML Universal Format)

**GGUF** is the primary file format used by `llama.cpp`. It is a single, self-contained binary format designed for efficient loading and sharing of models.

### GGUF File Structure:
```text
┌─────────────────────────────┐
│  GGUF File                  │
│  ├── Model metadata         │
│  │   (architecture, layers) │
│  ├── Tokenizer vocab        │
│  ├── Chat template          │
│  └── Weight tensors         │
│      (quantized or full)    │
└─────────────────────────────┘
```

### Why GGUF replaced GGML:
*   **Self-Describing:** Unlike the old GGML format, GGUF files contain all necessary metadata. You no longer need to pass model architecture types manually.
*   **Extensibility:** Supports arbitrary key-value metadata, allowing for future-proof updates.
*   **Compatibility:** Easier to introduce new model architectures without breaking existing implementations.

---

## Getting Started: Installation and Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### Step 2: Build the Project
Using CMake is the recommended way to build `llama.cpp`.

**Linux / macOS:**
```bash
cmake -B build
cmake --build build --config Release -j$(nproc)
```

**Windows:**
```powershell
cmake -B build
cmake --build build --config Release
```

### Step 3: Download a Model
Search for GGUF models on **HuggingFace** (e.g., "Mistral 7B GGUF"). Download the `.gguf` file to your `models/` directory.

### Step 4: Run Inference
```bash
./build/bin/llama-cli \
  -m models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  -p "What is the capital of France?" \
  -n 100
```
*   `-m`: Path to the model file.
*   `-p`: Your input prompt.
*   `-n`: Maximum number of tokens to generate.

---

## Technical Overview: The Inference Process

When you submit a prompt, `llama.cpp` performs the following steps:

1.  **Tokenization:** Converts raw text into a list of integer token IDs (e.g., `"Hello"` → `22557`).
2.  **Embedding Lookup:** Maps token IDs to high-dimensional vectors.
3.  **Transformer Layers:** Processes vectors through $N$ layers, involving:
    *   **Layer Norm**
    *   **Self-Attention:** Utilizing a **KV Cache** to store past keys and values for speed.
    *   **Feed-Forward Network (FFN)**
    *   **Residual Connections**
4.  **Output Head:** Converts the final hidden state into **logits** (scores for every possible token in the vocabulary).
5.  **Sampling:** Converts logits into probabilities and selects the next token based on parameters like `temperature`, `top-p`, or `top-k`.
6.  **Loop:** Repeats the process from step 3 until a stop token is generated or the limit is reached.

### Optimization Layer
`llama.cpp` implements these operations using:
*   **GGML:** Its own dedicated tensor library.
*   **CPU Optimization:** AVX2/AVX512 SIMD intrinsics.
*   **GPU Acceleration:** Specialized backends for NVIDIA (CUDA), Apple (Metal), and more.
```
