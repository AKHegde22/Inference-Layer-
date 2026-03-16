import requests
import math

def embed(text, port=8080):
    response = requests.post(
        f"http://localhost:{port}/v1/embeddings",
        json={"model": "local", "input": text}
    )
    return response.json()["data"][0]["embedding"]

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def semantic_search(query, documents, port=8080):
    query_vec = embed(query, port)
    results = []
    for doc in documents:
        doc_vec = embed(doc, port)
        score = cosine_sim(query_vec, doc_vec)
        results.append((score, doc))
    return sorted(results, key=lambda x: x[0], reverse=True)

docs = [
    "Python is a programming language",
    "The Eiffel Tower is in Paris",
    "Machine learning uses neural networks",
    "France is known for its cuisine"
]
results = semantic_search("AI and deep learning", docs)
for score, doc in results:
    print(f"{score:.3f}  {doc}")