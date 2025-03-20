# PDF RAG: Retrieval Augmented Generation

This project implements a **Retrieval Augmented Generation (RAG)** system that processes PDF documents, stores their content as embeddings in a Chroma vector database, and answers natural language queries using Hugging Face’s Inference API. Built with LangChain, it’s designed to be free, modular, and extensible—perfect for querying technical documents like the glibc heap manual.

## Features
- Loads PDFs from a "data" folder (e.g., glibc heap documentation).
- Splits documents into chunks and generates embeddings with "sentence-transformers/all-MiniLM-L6-v2".
- Stores embeddings in a persistent Chroma database.
- Answers queries (e.g., "What is the main topic?") using "mistralai/Mixtral-8x7B-Instruct-v0.1".
- Free to use with Hugging Face’s API (no OpenAI costs).

## Project Structure
- `main.py`: Runs the full pipeline—loads, processes, stores, and queries.
- `load_documents.py`: Loads PDFs using `UnstructuredPDFLoader`.
- `split_documents.py`: Splits text into chunks with unique IDs.
- `embeddings.py`: Generates embeddings via Hugging Face API.
- `database.py`: Manages the Chroma vector store.
- `query_app.py`: Retrieves relevant chunks and generates answers.
- `test_project.py`: Unit tests for all modules.

## Setup
### Prerequisites
- Python 3.8+
- Git
- A Hugging Face account for API access

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/0xGu4rdI4N/pdf_rag.git
   cd pdf_rag