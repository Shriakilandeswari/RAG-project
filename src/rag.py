import requests
import faiss
import numpy as np
from pypdf import PdfReader


ollama_url = "http://localhost:11434"

embed_model = "nomic-embed-text:latest"
llm_model = "llama3.2"

def load_book(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
    
def chuck_text(text, chunk_size=800):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
        return chunks
    
def get_embedding(text):
    response = requests.post(
        f"{ollama_url}/api/embed",
        json={"model":embed_model, "input": text}
    )
    print("response", response.json())
    # return 1
    return np.array(response.json()["embeddings"][0], dtype="float32")

def build_index(chunks):
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index

def search(query, index, chunks, k =3):
    query_emb = get_embedding(query).reshape(1,-1)
    distances, indices = index.search(query_emb, k)

    results = [chunks[i] for i in indices[0]]

    return results

def ask_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
    Answer ONLY using the context below.
If the answer is not in the context, say "I don't know.

    Context:{context}

    Question:{question}

    Answer:

"""
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={"model":llm_model, "prompt":prompt, "stream": False}
    )
    return response.json()["response"]


if __name__ == "__main__":
    book_text = load_pdf('../data/Essential-GraphRAG.pdf')
    chunks = chuck_text(book_text)

    print("Building vector index...")
    index = build_index(chunks)

    print("Ready!\n")

    while True:
        question = input("Ask about the book (or type 'exit'): ")
        if question.lower() == "exit":
            break

        retrieved = search(question, index, chunks)
        answer = ask_llm(question, retrieved)

        print("\nAnswer:\n", answer)
        print("\n" + "-"*50 + "\n")
