import os

import requests
import faiss
import numpy as np
import pickle

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2"

INDEX_FILE = os.path.join(os.path.dirname(__file__), "../index/book_index.faiss")
CHUNKS_FILE = os.path.join(os.path.dirname(__file__), "../index/chunks.pkl")


def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text}
    )
    data = response.json()

    if "embeddings" not in data:
        raise ValueError(f"Embedding error: {data}")

    return np.array(data["embeddings"], dtype="float32")


def search(query, index, chunks, k=3):
    query_emb = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_emb, k)
    print("indices",indices)
    return [chunks[i] for i in indices[0]]


def ask_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


if __name__ == "__main__":
    print("Loading index...")
    index = faiss.read_index(INDEX_FILE)

    print("Loading chunks...")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    print("Ready! Ask questions about the book.\n")

    while True:
        question = input("Ask (or type 'exit'): ")
        if question.lower() == "exit":
            break

        retrieved = search(question, index, chunks)
        answer = ask_llm(question, retrieved)

        print("\nAnswer:\n", answer)
        print("\n" + "-"*60 + "\n")