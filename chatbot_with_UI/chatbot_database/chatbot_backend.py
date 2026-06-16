import os
import sqlite3
import uuid
from typing import TypedDict, Annotated, Literal

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


def generate_thread_id():
    return str(uuid.uuid4())


load_dotenv()

loader = PyPDFLoader("intro_to_ml.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

search_tool = DuckDuckGoSearchRun()


@tool
def rag_tool(query):
    """Retrieve the related information from the pdf document.
        Use this tool when the user ask factual/conceptual question
        that might be answered from the stored document.
    """
    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    meta_data = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        # "metadata": meta_data,
    }


@tool
def calculator(
        first_num: float,
        second_num: float,
        operation: Literal["Add", "Subtract", "Multiply", "Divide"]
) -> dict:
    """ Perform arithmetic on two numbers. """
    match operation:
        case "Add":
            result = first_num + second_num
        case "Subtract":
            result = first_num - second_num
        case "Multiply":
            result = first_num * second_num
        case "Divide":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """ Get stock price for given symbol """
    API_KEY = os.environ.get("AlphaVantage_API_KEY")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url)
    return resp.json()


llm = ChatGroq(model="llama-3.3-70b-versatile", streaming=True)

tools = [search_tool, calculator, get_stock_price, rag_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']

    # Aggressively trim messages to stay well within Groq's 12,000 TPM limit.
    # We use character count (len) as a proxy (1 token ~ 4 chars).
    # 12,000 characters is roughly 3,000 tokens, providing ample headroom 
    # for tool definitions, system overhead, and the response.
    trimmed_messages = trim_messages(
        messages,
        strategy="last",
        token_counter=len,
        max_tokens=12000,
        start_on="human",
        include_system=True,
    )

    response = llm_with_tools.invoke(trimmed_messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
# Check Pointer
checkpointer = SqliteSaver(conn=conn)

# Creating our graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')
chatbot = graph.compile(checkpointer=checkpointer)

# response = chatbot.invoke(
#     {"messages": [
#         HumanMessage("What is the current stock price of Apple ? and How much it would cost of 50 shares ?")]},
#     config={"configurable": {"thread_id": "1"}},
# )
#
# print(response["messages"][-1].content)

while True:
    user_input = input("Ask Your Question: ")
    if user_input in ["q", "quit", "exit"]:
        break
    response = chatbot.invoke(
        {"messages": [HumanMessage(user_input)]},
        config={"configurable": {"thread_id": "1"}},
    )

    print("\nAnswer: ", response["messages"][-1].content, "\n")


def retrieve_all_thread():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

# png_data = chatbot.get_graph().draw_mermaid_png()
#
# with open("graph.png", "wb") as f:
#     f.write(png_data)
#
# print("Graph saved as graph.png")
