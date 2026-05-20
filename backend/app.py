# app.py
# Simple Chat UI using Streamlit + Ollama

import streamlit as st
import ollama

MODEL = "llama3"

st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Ollama Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        response_text = ""

        try:
            stream = ollama.chat(
                model=MODEL,
                messages=st.session_state.messages,
                stream=True
            )

            for chunk in stream:
                content = chunk["message"]["content"]
                response_text += content
                message_placeholder.markdown(response_text + "▌")

            message_placeholder.markdown(response_text)

        except Exception as e:
            response_text = f"Error: {e}"
            message_placeholder.error(response_text)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text
    })