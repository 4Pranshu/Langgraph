import streamlit as st
from langchain_core.messages import HumanMessage, AIMessageChunk

from chatbot_backend import chatbot, generate_thread_id, retrieve_all_thread


def reset_chat():
    generated_thread_id = generate_thread_id()
    st.session_state["thread_id"] = generated_thread_id
    add_thread(generated_thread_id)
    st.session_state["message_history"] = []


def add_thread(inp_thread_id: str):
    if inp_thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(inp_thread_id)


def load_conversation(thread_id):
    response = chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values
    if len(response) > 0:
        return response["messages"]
    else:
        return []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_thread()

add_thread(st.session_state["thread_id"])

CONFIG = {
    "configurable": {
        "thread_id": st.session_state["thread_id"],
    },
    "run_name": "Streamlit Chatbot",
    "metadata": {
        "session_id": st.session_state["thread_id"],  # <-- this is what powers Threads tab
    },
}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# -------------------------------------------- Sidebar UI -----------------------------------------
st.sidebar.title("ChatBot Streamlit")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversation")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(thread_id):
        if st.session_state["thread_id"] != thread_id:
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = "user"
                else:
                    role = "ai"
                temp_messages.append({"role": role, "message": msg.content})

            st.session_state["message_history"] = temp_messages

# Display chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["message"])

user_input = st.chat_input("Type your message here")

if user_input:
    st.session_state["message_history"].append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.text(user_input)


    # Stream AI response
    def stream_response():
        for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(user_input)]},
                config=CONFIG,
                stream_mode="messages",
        ):
            if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                yield message_chunk.content


    with st.chat_message("ai"):
        ai_message = st.write_stream(stream_response())

    st.session_state["message_history"].append({"role": "ai", "message": ai_message})
