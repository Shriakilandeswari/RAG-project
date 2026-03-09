RAG Chatbot using FAISS and Ollama

This project implements a Retrieval-Augmented Generation (RAG) system that allows a language model to answer questions using external documents. It combines:

FAISS for vector search

Ollama for LLM embeddings

LangChain for orchestration

Features

PDF document ingestion

Text chunking for embeddings

Vector database search with FAISS

Interactive chat interface

Project Structure
rag-project/
│
├─ src/ # Core RAG logic (build_index.py, chat.py)
├─ data/ # Source documents (PDFs)
├─ index/ # FAISS vector database & pickled chunks
└─ main.py # Program entry point with menu (build or chat)
Setup

Clone the repository

git clone https://github.com/yourusername/rag-project.git
cd rag-project

Create a virtual environment

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies

pip install -r requirements.txt

Build the index and/or start chat

Run the menu-based entry point:

python main.py

You will see options:

1. Build Index
2. Start Chat

Build Index → Reads PDFs, chunks text, generates embeddings, and saves FAISS index.

Start Chat → Loads the FAISS index and allows you to query the document interactively.

Example Usage

After building the index, start the chat and ask questions like:

What is GraphRAG?
Explain the key concepts in the document.

The bot will return the most relevant text chunks from your PDFs.

Notes

Ensure the Ollama server is running locally on port 11434 for embeddings:

ollama serve

Index files and chunks are stored in the index/ folder.

PDFs should be placed in the data/ folder.

Technologies

Python

FAISS (Vector Search)

LangChain (Orchestration)

Ollama (Embeddings and LLM)
