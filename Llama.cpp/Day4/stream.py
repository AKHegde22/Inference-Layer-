from openai import OpenAI

def stream_chat(prompt, system="You are helpful.", port=8080):
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="none"
    )
    stream = client.chat.completions.create(
        model="local",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
        stream=True
    )
    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_response += delta
    print()
    return full_response

response = stream_chat("Explain the KV cache in 3 sentences")
print(f"\nFull response length: {len(response)} chars")
