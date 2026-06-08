import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

message_history = st.session_state["message_history"]

for message in message_history:
    with st.chat_message(message["role"]):
        st.text(message["message"])

user_input = st.chat_input("Type your message here")

if user_input:
    message_history.append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    response = chatbot.invoke({"messages": [HumanMessage(user_input)]}, config={"configurable": {"thread_id": "1"}})
    answer = response["messages"][-1].content
    message_history.append({"role": "ai", "message": answer})
    with st.chat_message("ai"):
        st.text(answer)

    st.session_state["message_history"] = message_history
