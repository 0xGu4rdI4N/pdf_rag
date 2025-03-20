def query_rag(query, vectorstore, llm, k=5):
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Use the following context to answer the question concisely.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm.invoke(prompt)  # Use invoke instead of __call__ to avoid deprecation
    return response