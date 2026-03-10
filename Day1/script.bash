1."Write the two CMake commands needed to build llama.cpp on Linux/macOS. Use all available CPU cores for the build."

cmake -B build
cmake --build build --config Release -j$(nproc)

2."Writing a llama-cli command that:
• Loads model at: models/mistral.Q4_K_M.gguf
• Uses the prompt: "Explain attention in transformers"
• Generates a maximum of 200 tokens
• Uses 8 threads"

../build/bin/llama-cli \
  -m models/mistral.Q4_K_M.gguf \
  -p "Explain attention in transformers" \
  -n 200 \
  -t 8


3."Writing a Python snippet using the openai SDK to send a chat message "What is GGUF?" to a local llama-server running on port 8080. Print the response content."

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "What is GGUF?"}]
)

print(response.choices[0].message.content)