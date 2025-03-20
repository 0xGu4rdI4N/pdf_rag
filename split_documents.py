from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks with unique identifiers.
    
    Args:
        documents (List[Document]): List of Document objects to split.
        chunk_size (int): Maximum size of each chunk in characters.
        chunk_overlap (int): Number of overlapping characters between chunks.
    
    Returns:
        List[Document]: List of split Document objects with unique IDs in metadata.
    """
    # Initialize the text splitter with recursive character-based splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    # Split documents into chunks, preserving metadata
    split_docs = text_splitter.split_documents(documents)
    
    # Assign unique IDs to each chunk based on source, page, and chunk index
    chunk_counter = defaultdict(int)
    for doc in split_docs:
        source = doc.metadata["source"]
        page = doc.metadata.get("page", 0)  # Default to 0 if page is missing
        chunk_index = chunk_counter[(source, page)]
        doc.metadata["chunk_index"] = chunk_index
        doc.metadata["id"] = f"{source}_{page}_{chunk_index}"
        chunk_counter[(source, page)] += 1
    
    return split_docs