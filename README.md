# RAG Chatbot using FAISS and Ollama

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows a language model to answer questions using external documents.

## Features

- Document chunking
- Vector embeddings
- FAISS vector search
- Chat interface for querying documents

## Project Structure

- `src/` → core RAG logic
- `data/` → source documents
- `index/` → vector database
- `main.py` → program entry point

## Setup

Clone the repository

```
git clone https://github.com/yourusername/rag-project.git
cd rag-project
```

Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

Build the index

```
python main.py
```

## Technologies

- Python
- FAISS
- LangChain
- Ollama

## Example

Ask questions about the document:

```
What is GraphRAG?
```
