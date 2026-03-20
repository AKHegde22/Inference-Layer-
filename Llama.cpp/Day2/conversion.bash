# Command 1: Convert to GGUF
python convert_hf_to_gguf.py ./my-llama-model \
  --outtype f16 \
  --outfile ./models/llama-f16.gguf

# Command 2: Quantize
./build/bin/llama-quantize \
  ./models/llama-f16.gguf \
  ./models/llama-q5_k_m.gguf \
  Q5_K_M

# Command 3: Test
./build/bin/llama-cli \
  -m ./models/llama-q5_k_m.gguf \
  -p "Hello" -n 50