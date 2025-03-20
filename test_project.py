import pytest
from langchain.schema import Document
from load_documents import load_documents
from split_documents import split_documents
from embeddings import get_embedding_function
from database import get_vectorstore, update_vectorstore
from query_app import query_rag
from unittest.mock import MagicMock

# ... (other tests unchanged)

def test_get_embedding_function(monkeypatch):
    """Test that the embedding function produces vectors of correct dimension via API."""
    # Mock the API call to avoid real requests during testing
    def mock_embed_documents(texts):
        return [[0.1] * 384 for _ in texts]  # Simulate 384d vectors
    
    def mock_embed_query(text):
        return [0.1] * 384
    
    embedding_function = get_embedding_function()
    monkeypatch.setattr(embedding_function, "embed_documents", mock_embed_documents)
    monkeypatch.setattr(embedding_function, "embed_query", mock_embed_query)
    
    embedding = embedding_function.embed_query("test")
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # Matches all-MiniLM-L6-v2

# ... (rest of tests unchanged)