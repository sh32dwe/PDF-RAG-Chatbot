# PDFQueryBot - RAG-based PDF Question Answering System

## 🚀 Overview
PDFQueryBot is an intelligent question-answering system that leverages Retrieval Augmented Generation (RAG) to provide accurate, context-aware answers from PDF documents. Built with Llama-2 and LangChain, it combines the power of large language models with efficient document retrieval to make your PDFs interactive and queryable.

## ✨ Key Features

- **Semantic Search**: Utilizes FAISS vector store for efficient similarity search
- **Smart Text Processing**: Automatic PDF parsing and intelligent text chunking
- **Context-Aware Responses**: Uses Llama-2 7B model to generate relevant answers
- **Interactive Interface**: Simple command-line interface for easy interaction
- **Customizable**: Adjustable prompt templates and retrieval parameters

## 🛠️ Technical Stack

- **Large Language Model**: Llama-2 7B
- **Framework**: LangChain
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Store**: FAISS
- **Document Processing**: PyPDF

## 📋 Prerequisites

- Python 3.8+
- Hugging Face account and API token
- 16GB+ RAM (recommended)
- GPU (recommended for faster inference)