from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

TEMPLATE = """
    Sistema
    debes responder siempre en español
    debes responder de manera corta y concisa
    debes responder de manera amigable
    no puedes responder cosas ofensivas, ni temas sensibles
    Tu eres un chatbot que debe responder de manera corta y concisa a la pregunta:
"""

messages = [
]

llm = ChatOllama(model="llama3")
prompt = ChatPromptTemplate.from_messages(template_format=TEMPLATE, messages=messages)

def chatHistory(prompt):
    if len(messages) > 10:
        messages.pop(0)
    messages.append(HumanMessage(content=prompt))
    return messages


def getAnswer(question):
    chat = chatHistory(question)
    answer = llm.invoke(chat)
    return answer.content
    



