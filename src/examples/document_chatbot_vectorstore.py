import os

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def create_vectorstore(file_path, vectorstore_path) -> FAISS | None:
    # Esta función crea un vectorstore a partir de un archivo PDF.
    # Usaremos PyPDFLoader para cargar el contenido del PDF y FAISS para crear el vectorstore.
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    faiss_index.save_local(vectorstore_path)

    return faiss_index


def load_vectorstore(vectorstore_path):
    # Esta función carga un vectorstore desde un archivo usando FAISS
    # Más info aquí https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/
    faiss_index = FAISS.load_local(
        vectorstore_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True
    )
    return faiss_index


def get_or_create_vectorstore(file_path, vectorstore_path):
    # Esta función carga un vectorstore si ya existe o lo crea si no.
    if os.path.exists(vectorstore_path):
        vectorstore = load_vectorstore(vectorstore_path)
    else:
        vectorstore = create_vectorstore(file_path, vectorstore_path)
    return vectorstore
