from langchain_chroma import Chroma
# Rest of your code remains the same
def get_vectorstore(persist_directory="db", embedding_function=None):
    """
    Initialize or load a Chroma vector store.
    
    Args:
        persist_directory (str): Directory for persistent storage of the vector database.
        embedding_function: Function to embed text (required for consistency).
    
    Returns:
        Chroma: A Chroma vector store instance.
    """
    # Load existing database or create a new one if it doesn’t exist
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    return vectorstore

def update_vectorstore(vectorstore, new_docs):
    """
    Update the vector store with new documents, avoiding duplicates.
    
    Args:
        vectorstore (Chroma): The Chroma vector store instance to update.
        new_docs (List[Document]): List of new Document objects to add.
    """
    # Get existing IDs to prevent duplicates
    existing_ids = set(vectorstore.get()['ids'])
    new_docs_to_add = [doc for doc in new_docs if doc.metadata["id"] not in existing_ids]
    
    # Add new documents if there are any
    if new_docs_to_add:
        new_ids = [doc.metadata["id"] for doc in new_docs_to_add]
        vectorstore.add_documents(new_docs_to_add, ids=new_ids)
        vectorstore.persist()  # Save changes to disk