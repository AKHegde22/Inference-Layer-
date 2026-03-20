import json
import requests

def extract_structured(text, schema, port=8080):
    keys = ", ".join(f'"{k}" ({v})' for k, v in schema.items())
    system = f"Extract information as JSON with these fields: {keys}. Return only valid JSON, no extra text."
    
    response = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": text}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 300
        }
    )
    content = response.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Model returned invalid JSON: {content}")

def extract_person(text, port=8080):
    schema = {"name": "str", "birth_year": "int",
              "nationality": "str", "field": "str"}
    return extract_structured(text, schema, port)

result = extract_person(
    "Marie Curie was a Polish-French physicist born in 1867."
)
print(result)