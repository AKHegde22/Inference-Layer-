./build/bin/llama-server \
  -m models/llama3-8b.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8000 \
  -ngl 99 \
  --flash-attn \
  --parallel 4 \
  --cont-batching \
  -c 8192 \
  -a "llama3"
