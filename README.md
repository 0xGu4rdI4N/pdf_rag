# Retrieval Augmented Generation (RAG) Project

This project implements a Retrieval Augmented Generation (RAG) system using LangChain and OpenAI's GPT-3.5 API. It processes PDF documents from a "data" folder, splits them into chunks, stores embeddings in a Chroma vector database, and generates answers to natural language queries.

## Project Structure

- `load_documents.py`: Loads PDFs from "data".
- `split_documents.py`: Splits documents into chunks.
- `embeddings.py`: Provides embedding function with SentenceTransformers.
- `database.py`: Manages Chroma vector store.
- `query_app.py`: Queries the system and generates answers.
- `test_project.py`: Unit tests for all modules.
- `main.py`: Runs the full pipeline.

## Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate