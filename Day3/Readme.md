# Temperature & Randomness

**TEMPERATURE** — controls how "creative" or "random" the model is.

Technically, temperature scales the logits before the softmax function is applied:
```python
scaled_logit = logit / temperature
```

- **`temp = 0.0`**: Greedy (always pick highest probability token). Deterministic, repetitive, factual.
- **`temp = 0.1`**: Near-deterministic, great for code/math.
- **`temp = 0.7`**: Balanced creativity (recommended default).
- **`temp = 1.0`**: Raw model probabilities, no scaling.
- **`temp = 1.5`**: Very creative, sometimes incoherent.
- **`temp = 2.0+`**: Near-random, usually gibberish.

**Example: Prompt "The sky is..."**
- `temp=0.1` → "blue." (always)
- `temp=0.7` → "blue." / "a canvas of light."
- `temp=1.5` → "blue." / "weeping tonight" / "full of cats"

**Usage Examples:**
```bash
./build/bin/llama-cli -m model.gguf --temp 0.7 -p "Your prompt"

# For code generation (deterministic):
./build/bin/llama-cli -m model.gguf --temp 0.1 -p "Write a sort function"

# For creative writing:
./build/bin/llama-cli -m model.gguf --temp 0.9 -p "Write a poem"
```

---

# Top-K, Top-P, Min-P Sampling

These parameters filter the token candidates **BEFORE** sampling. They work together with temperature.

### TOP-K (`-top-k`)
Only consider the top K most likely tokens.
- `top-k=1`: Always pick #1 token (greedy)
- `top-k=40`: Pick from top 40 candidates (default)
- `top-k=0`: Disabled (consider all tokens)

*Problem:* A fixed K ignores probability distribution shape. If top-2 tokens have 95% probability, why allow 38 more?

### TOP-P / NUCLEUS SAMPLING (`-top-p`)
Pick from the smallest set of tokens whose cumulative probability ≥ P.
- `top-p=0.9`: Keep adding tokens until sum of probs ≥ 90%
- `top-p=1.0`: All tokens (disabled)
- `top-p=0.5`: Very focused, only highest-prob tokens

*Example with `top-p=0.9`:*
- Token A: 60%  ┐ 
- Token B: 20%  │ Keep 
- Token C: 10%  ┤ cumsum = 90% → STOP here
- Token D:  5%  │ ← excluded
- ...           ┘

### MIN-P (`-min-p`) 
*Newer, and often better.*
Exclude any token whose probability < (`top_prob` × `min_p`)
- `min-p=0.05`: Drop tokens less than 5% of top token's prob

*Why it's smarter:* Scales relative to the current distribution. If top token is 80% probable, min threshold = `80% × 0.05 = 4%`.

### Recommended Combos:
```bash
# Balanced chat
--temp 0.7 --top-k 40 --top-p 0.9

# Creative writing  
--temp 0.9 --top-p 0.95 --min-p 0.05

# Code / factual
--temp 0.1 --top-k 10 --top-p 0.9
```

---

# Repeat Penalty & Context

### REPEAT PENALTY (`--repeat-penalty`)
Penalizes tokens that have already appeared in the output.
- `penalty=1.0`: No penalty (disabled)
- `penalty=1.1`: Mild penalty (recommended default)
- `penalty=1.3`: Strong penalty
- `penalty=1.5`: Very strong, may degrade quality

*How it works:*
```python
new_logit = logit / repeat_penalty   # if token appeared before
new_logit = logit                    # if token is new
```

**Related flags:**
- `--repeat-last-n N`: Look back N tokens for repeats (-1 = entire context, default=64)
- `--presence-penalty`: Flat penalty for any prior token
- `--frequency-penalty`: Scales with how often token appeared

### CONTEXT WINDOW (`-c`)
Maximum number of tokens the model can "see" at once. Includes both your input AND the generated output.
- `-c 2048`: 2K context (small, fast)
- `-c 4096`: 4K context (default for many models)
- `-c 8192`: 8K context
- `-c 32768`: 32K context (needs lots of RAM — KV cache!)

*KV Cache memory grows with context:*
```python
kv_cache_bytes ≈ 2 * layers * heads * head_dim * ctx_len * 2
```
- For Mistral 7B at 4K ctx ≈ ~500 MB
- For Mistral 7B at 32K ctx ≈ ~4 GB ← huge!

### BATCH SIZE & THREADS (`-b`, `-t`)
- `-b 512`: Prompt processing batch size (higher = faster prompt eval)
- `-t 8`: CPU threads for inference (match your physical core count)
- `-tb 4`: Separate thread count for batch processing

---

# Prompt Formatting & Chat Templates

Chat models are fine-tuned with special tokens wrapping each message. If you don't use the correct format, quality degrades significantly.

### LLAMA 3 FORMAT
```text
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is 2+2?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

### MISTRAL / CHATML FORMAT
```text
<s>[INST] You are helpful. What is 2+2? [/INST]
4</s>
[INST] And 3+3? [/INST]
```

### CHATML FORMAT (Qwen, many others)
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```

### EASIEST: USE `-cnv` (conversation mode)
`llama-cli` reads the chat template from the GGUF file and applies it automatically!

```bash
./build/bin/llama-cli \
  -m model.gguf \
  -cnv \
  --sys "You are a Python expert."

# Or use llama-server (handles templates automatically)
```
