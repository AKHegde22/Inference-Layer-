import math, requests, json

def embed(text, port=8080):
    r = requests.post(f"http://localhost:{port}/v1/embeddings",
                      json={"model": "local", "input": text})
    return r.json()["data"][0]["embedding"]

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

class RAG:
    def __init__(self, embed_port=8080, chat_port=8080):
        self.embed_port = embed_port
        self.chat_port = chat_port
        self.docs = []

    def add_document(self, doc_id, text):
        vec = embed(text, self.embed_port)
        self.docs.append({"id": doc_id, "text": text, "embedding": vec})

    def retrieve(self, query, top_k=3):
        q_vec = embed(query, self.embed_port)
        scored = sorted(self.docs,
                        key=lambda d: cosine_sim(q_vec, d["embedding"]),
                        reverse=True)
        return [d["text"] for d in scored[:top_k]]

    def answer(self, question, top_k=3):
        context = "\n".join(self.retrieve(question, top_k))
        prompt = f"Answer based on context:\n{context}\n\nQuestion: {question}"
        r = requests.post(
            f"http://localhost:{self.chat_port}/v1/chat/completions",
            json={"model": "local",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 300}
        )
        return r.json()["choices"][0]["message"]["content"]

rag = RAG()
rag.add_document("d1", "llama.cpp runs LLMs locally on CPU and GPU.")
rag.add_document("d2", "GGUF is the file format used by llama.cpp for storing models.")
rag.add_document("d3", "Quantization reduces model size by lowering weight precision.")
rag.add_document("d4", "The Eiffel Tower is located in Paris, France.")
print(rag.answer("What file format does llama.cpp use?"))
