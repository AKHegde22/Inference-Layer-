from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import json

class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    industry: str
    founded_year: int
    headquarters: Address
    employee_count: Optional[int] = None
    public: bool = False

def extract_company(text, port=8000):
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
    response = client.chat.completions.create(
        model="local",
        messages=[
            {"role": "system", "content": "Extract company information as JSON. Return only JSON, no extra text."},
            {"role": "user",   "content": text}
        ],
        extra_body={"guided_json": Company.model_json_schema()},
        temperature=0.0,
        max_tokens=400,
    )
    content = response.choices[0].message.content
    try:
        return Company.model_validate_json(content)
    except Exception:
        data = json.loads(content)
        return Company(**data)

def batch_extract(texts, port=8000):
    return [extract_company(t, port) for t in texts]

texts = [
    "OpenAI was founded in 2015, is based at 3180 18th St, San Francisco, USA. "
    "It's a private AI company with around 1000 employees.",
    "Apple Inc. is a public tech company founded in 1976, headquartered at "
    "One Apple Park Way, Cupertino, USA with over 160000 employees.",
    "Mistral AI, founded in 2023, is a private French AI startup at "
    "15 Rue des Halles, Paris, France.",
]

companies = batch_extract(texts)
for c in companies:
    print(f"  {c.name:<20} | {c.industry:<12} | {c.headquarters.city}")
