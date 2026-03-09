import os

import requests
import faiss
import numpy as np
import pickle
from pypdf import PdfReader

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"

PDF_FILE = os.path.join(os.path.dirname(__file__), "../data/Essential-GraphRAG.pdf")
INDEX_FILE = os.path.join(os.path.dirname(__file__), "../index/book_index.faiss")
CHUNKS_FILE = os.path.join(os.path.dirname(__file__), "../index/chunks.pkl")


def load_pdf(path):
    reader = PdfReader(path)
    print("reader", reader)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=800):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text}
    )
    data = response.json()
    print("data", data)

    if "embeddings" not in data:
        raise ValueError(f"Embedding error: {data}")

    return np.array(data["embeddings"], dtype="float32")


def build_index(chunks):
    embeddings = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}")
        embeddings.append(get_embedding(chunk))

    embeddings = np.vstack(embeddings)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


if __name__ == "__main__":
    print("Reading PDF...")
    text = load_pdf(PDF_FILE)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Building FAISS index...")
    index = build_index(chunks)

    print("Saving index...")
    faiss.write_index(index, INDEX_FILE)

    print("Saving chunks...")
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print("Done. Index stored successfully.")