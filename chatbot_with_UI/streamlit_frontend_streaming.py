import streamlit as st
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph_backend import chatbot

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["message"])

user_input = st.chat_input("Type your message here")

if user_input:
    # Show user message
    st.session_state["message_history"].append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Stream AI response
    def stream_response():
        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(user_input)]},
            config={"configurable": {"thread_id": "1"}},
            stream_mode="messages",
        ):
            if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                yield message_chunk.content

    with st.chat_message("ai"):
        ai_message = st.write_stream(stream_response())

    st.session_state["message_history"].append({"role": "ai", "message": ai_message})