from examples.document_chatbot_vectorstore import get_or_create_vectorstore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Inicializamos el modelo de chat (Usaremos OpenAI para el ejemplo porque es el
# más popular y fácil de usar)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# Cargamos el vectorstore que contiene el contenido del libro o creamos uno si
# no existe
faiss_index = get_or_create_vectorstore(
    "./files/book.pdf", "./files/book_vectorstore.faiss")

# Y lo convertimos en un retriever para poder usarlo en el chatbot
faiss_index_retriever = faiss_index.as_retriever()

# Creamos una plantilla para el chatbot
TEMPLATE = """
Eres un asustente para tareas de pregunta y respuesta. Usa el siguiente contexto
recuperado para responder la pregunta. Si no sabes la respuesta, simplemente di
que no sabes. Usa tres oraciones como máximo y mantén la respuesta concisa.

Pregunta: {question} 

Contexto: {context} 

Respuesta:
"""

# Creamos el prompt a partir de la plantilla
prompt = PromptTemplate.from_template(template=TEMPLATE)


def format_docs(docs):
    # Esta función formatea los documentos para que el chatbot pueda entenderlos
    # como texto plano
    return "\n\n".join(doc.page_content for doc in docs)


# Creamos el rag por partes: primero preparando el contexto y la pregunta. Para
# el contexto usamos el retriever y lo pasamos por la función que formatea los
# documentos, para la pregunta usamos RunnablePassthrough para pasarla tal cual
# al chatbot cuando lo invoquemos. Luego pasamos el prompt, el chatbot y el
# parser de salida. Esto es un ejemplo de uso de LCEL (LangChain Expression
# Language, más info aquí: https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel)
# para unir todo.
rag_chatbot = (
    {"context": faiss_index_retriever | format_docs,
        "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
