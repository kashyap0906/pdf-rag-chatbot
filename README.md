# 📄 Local RAG PDF Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built in Python that can answer questions from a PDF document using local AI models.

## 🚀 Features

- Read text from PDF files
- Split documents into chunks
- Generate embeddings using Sentence Transformers
- Store embeddings using FAISS
- Retrieve the most relevant document chunks
- Generate answers using a local LLM (Qwen via Ollama)
- Runs completely offline (after the model is downloaded)

## 🛠️ Tech Stack

- Python
- PyPDF
- LangChain Text Splitters
- Sentence Transformers
- FAISS
- Ollama
- Qwen 2.5 1.5B

## 📂 Project Structure

```
.
├── app.py
├── README.md
├── .gitignore
├── data/
│   └── sample.pdf
└── venv/ (ignored)
```

## ⚙️