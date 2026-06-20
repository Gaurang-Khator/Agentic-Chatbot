from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List


class AgentState(TypedDict):
    messages : Annotated[List[BaseMessage], add_messages]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def chat_node(state: AgentState) -> AgentState:

    messages = state['messages']

    response = llm.invoke(messages)

    return { "messages" : response}


graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm", chat_node)

graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)

graph = graph_builder.compile()

print("\n ===== AGENTIC-CHATBOT ===== \n")

user_input = input("YOU : ")

while user_input not in ['exit', 'quit', 'bye']:

    result = graph.invoke({"messages" : HumanMessage(content=user_input)})

    print(f"\nAI : {result['messages'][-1].content}\n")

    user_input = input("YOU : ")

print("\n ===== THANK YOU FOR USING AGENTIC-CHATBOT ===== \n")
