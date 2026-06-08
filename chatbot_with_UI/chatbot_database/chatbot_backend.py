import sqlite3
import uuid
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


def generate_thread_id():
    return str(uuid.uuid4())


load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", streaming=True)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect("./chatbot_database/chatbot.db", check_same_thread=False)
# Check Pointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# response = chatbot.invoke(
#     {"messages": [HumanMessage("What is my name ?")]},
#     config={"configurable": {"thread_id": "1"}},
# )
#
# print(response)
def retrieve_all_thread():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
