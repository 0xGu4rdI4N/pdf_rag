import os
from langchain_community.llms import HuggingFaceHub
from load_documents import load_documents
from split_documents import split_documents
from embeddings import get_embedding_function
from database import get_vectorstore, update_vectorstore
from query_app import query_rag

def main():
    embedding_function = get_embedding_function()
    documents = load_documents()
    split_docs = split_documents(documents)
    vectorstore = get_vectorstore(persist_directory="db", embedding_function=embedding_function)
    update_vectorstore(vectorstore, split_docs)
    
    hf_api_token = os.getenv("HF_API_TOKEN")
    if not hf_api_token:
        raise ValueError("HF_API_TOKEN not set. Get it from https://huggingface.co/settings/tokens.")
    
    # Initialize HuggingFaceHub directly
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=hf_api_token,
        model_kwargs={"temperature": 0.1}
    )
    
    query = "What is the main topic of the documents?"
    answer = query_rag(query, vectorstore, llm)
    print(f"Query: {query}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()