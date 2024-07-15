from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def create_vectorstore(file_path, vectorstore_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    faiss_index.save_local(vectorstore_path)


def load_vectorstore(vectorstore_path):
    faiss_index = FAISS.load_local(
        vectorstore_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True
    )
    return faiss_index
