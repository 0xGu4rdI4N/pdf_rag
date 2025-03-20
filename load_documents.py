import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# Rest of your code remains the same
def load_documents(data_dir="data"):
    """
    Load PDF documents from the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing PDF files.
    
    Returns:
        List[Document]: List of Document objects, each representing a page from the PDFs.
    
    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If no PDF files are found in the directory.
    """
    # Check if the directory exists to prevent invalid path errors
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")
    
    # Use DirectoryLoader to load all PDFs with PyPDFLoader for text extraction
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Ensure at least one document is loaded
    if not documents:
        raise ValueError(f"No PDF files found in '{data_dir}'.")
    
    return documents