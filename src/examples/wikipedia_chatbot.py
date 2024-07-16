from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Inicializamos el modelo de chat (Usaremos OpenAI para el ejemplo porque es el
# más popular y fácil de usar)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# Configuraremos la herramienta de Wikipedia para que el chatbot pueda buscar
# información en Wikipedia.
# Sus componentes son un wrapper de la API de Wikipedia que se encarga de
# busca el mejor resultado hasta un límite de 2000 caracteres.
wikipedia_api_wrapper = WikipediaAPIWrapper(
    top_k_results=1, doc_content_chars_max=2000)
# Y la herramienta en sí misma, que se encarga de buscar en Wikipedia.
wikipedia_tool = WikipediaQueryRun(
    name="Wikipedia",
    description="Search using Wikipedia",
    api_wrapper=wikipedia_api_wrapper,
)
tools = [wikipedia_tool]

# Creamos nuestra plantilla inicial para que el chatbot sepa cómo responder
TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer. This is in Spanish. Translate to English for the Thought.

Thought: you should always think about what to do. In English.

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action. In English.

Observation: the result of the action. In English.

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question. Only this has to be in Spanish. Your answers should be lengthy, not just a one-sentence summary.

Begin!

Question: {input}

Thought: {agent_scratchpad}"""

# Creamos nuestro prompt para el chat a partir de la plantilla
prompt = ChatPromptTemplate.from_template(template=TEMPLATE)

# Creamos el agente usando el modelo ReAct (https://react-lm.github.io/)
agent = create_react_agent(llm, tools, prompt)

# Creamos el ejecutor usando todo lo anterior.
# Nota: este concepto será eventualmente migrado a LangGraph (https://langchain-ai.github.io/langgraph/)
wikipedia_chatbot = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)
