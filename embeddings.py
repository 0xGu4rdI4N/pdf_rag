import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# Rest of your code remains the same    
def get_embedding_function():
    """
    Provide an embedding function using Hugging Face's Inference API.
    
    Returns:
        HuggingFaceInferenceAPIEmbeddings: An embedding function using all-MiniLM-L6-v2 via API.
    
    Raises:
        ValueError: If HF_API_TOKEN is not set in environment variables.
    """
    api_key = os.getenv("HF_API_TOKEN")
    if not api_key:
        raise ValueError("HF_API_TOKEN environment variable not set. Get it from https://huggingface.co/settings/tokens.")
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )