from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

TEMPLATE = """
    Tu eres un chatbot que debe responder de manera corta y concisa a la pregunta:
    {question}
    reglas:
    debes responder siempre en español
    debes responder de manera corta y concisa
    debes responder de manera amigable
    no puedes responder cosas ofensivas, ni temas sensibles
"""

def initModel():
    llm = ChatOllama(model="llama3")
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    return chain

def getAnswer(question):
    chain = initModel()
    return chain.invoke({"question": question})

def main():
    question = "¿Como murio pablo escobar?"
    answer = getAnswer(question)
    print(answer)

main()

