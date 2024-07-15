from examples.document_chatbot_vectorstore import load_vectorstore

faiss_index = load_vectorstore("vectorstore.faiss")
docs = faiss_index.similarity_search("What is chaos?", k=40)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
