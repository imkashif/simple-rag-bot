# 🤖 Lightweight Mini-RAG Bot

A fast, fully local Retrieval-Augmented Generation (RAG) engine built from scratch. This project allows users to chat with their personal `.txt` and `.pdf` documents using either an interactive terminal console or a Telegram Bot interface.

Unlike heavy frameworks (like LangChain or LlamaIndex) or expensive cloud vector databases (like Pinecone), this project uses a highly optimized, custom-built SQLite + Numpy vector search pipeline to keep the application lightweight, blazing fast, and free to host.

## ✨ Key Features
* **Multi-Format Document Parsing:** Automatically reads, chunks, and extracts text from standard `.txt` files and `.pdf` documents using `PyPDF2`.
* **100% Local Embeddings:** Uses Hugging Face's `all-MiniLM-L6-v2` (`sentence-transformers`) to generate vector embeddings entirely on your local machine, saving API costs.
* **Optimized Vector Search:** * Implements **Batch Embedding** to drastically reduce the time it takes to process hundreds of documents.
  * Uses an **In-Memory Numpy Matrix** to perform instantaneous Cosine Similarity mathematical routing across all document vectors.
* **Dual Interfaces:** Comes with a lightning-fast Terminal Console loop for local testing and a fully functional Telegram Bot deployment script.
* **Source Citations:** The bot explicitly lists the exact document snippets and confidence scores it used to generate its answers to prevent hallucinations.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **LLM Generation:** OpenAI API (`gpt-3.5-turbo` or `gpt-4o`)
* **Local Embedding:** `sentence-transformers`
* **Vector Database:** Built from scratch using `sqlite3` + `numpy`
* **Document Parsing:** `PyPDF2`
* **Bot Interface:** `simple-rag-bot`

## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
cd your-repo-name

