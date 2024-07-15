from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# Inicializamos el modelo de chat (Usaremos OpenAI para el ejemplo porque es el más popular y fácil de usar)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
)

# Creamos nuestra memoria de mensajes. Usaremos Streamlit para mostrar los mensajes en la interfaz de usuario.
history = ChatMessageHistory()

# Creamos nuestra plantilla inicial para que el chatbot sepa cómo responder
TEMPLATE = """
Eres un experto en interpretar el corazón y el espíritu de una pregunta y responder de manera perspicaz.

PASOS
1. Comprende profundamente lo que se pregunta.
2. Crea un modelo mental completo de la entrada y la pregunta en una pizarra virtual en tu mente.
3. Responde a la pregunta en 3-5 viñetas.

INSTRUCCIONES
1. Solo escribe viñetas para responder las preguntas. Nada extra.
2. No titules tu respuesta.
3. No respondas con el modelo mental o la pizarra virtual.
4. No produzcas advertencias ni notas, solo las secciones solicitadas.
5. Si el usuario saluda, responde con un saludo en texto libre únicamente.
6. Tus respuestas serán siempre en español.
"""

# Creamos nuestro prompt de chat
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(TEMPLATE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Encadenamos el prompt con el modelo de chat usando el Langchain Expression
chain = prompt | llm

# Agregamos el historial de mensajes a la cadena
chatbot = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="question",
    history_messages_key="history",
)
