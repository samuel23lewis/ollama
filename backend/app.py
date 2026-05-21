# app.py
# Streamlit + Ollama chatbot

import streamlit as st
import ollama

MODEL_OPTIONS = [
    "llama3",
    "gemma:2b",
    "gemma:latest",
]


def build_export_text(messages):
    lines = []

    for msg in messages:
        role = msg["role"].capitalize()
        content = (msg.get("content") or "").strip()

        if content:
            lines.append(f"{role}: {content}")
            lines.append("")

    return "\n".join(lines)


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="O",
    layout="centered",
)

# ---------------- SESSION STATE ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- TITLE ---------------- #
st.title("Ollama Chatbot")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Settings")

MODEL = st.sidebar.selectbox(
    "Choose Model",
    MODEL_OPTIONS,
)

# ---------------- CLEAR CHAT BUTTON ---------------- #
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ---------------- EXPORT CHAT ---------------- #
st.sidebar.download_button(
    label="Export Chat",
    data=build_export_text(st.session_state.messages),
    file_name="chat_history.txt",
    mime="text/plain",
)

# ---------------- DISPLAY CHAT HISTORY ---------------- #
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ---------------- #
prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_text = ""

        try:
            stream = ollama.chat(
                model=MODEL,
                messages=st.session_state.messages,
                stream=True,
            )

            for chunk in stream:
                content = chunk["message"]["content"]
                response_text += content
                message_placeholder.markdown(response_text + "|")

            message_placeholder.markdown(response_text)

        except Exception as e:
            response_text = f"""
[Error] Error connecting to Ollama

Make sure:
- Ollama is installed
- Ollama is running
- The selected model `{MODEL}` is pulled

Example:
`ollama pull {MODEL}`

Technical Error:
{e}
""".strip()

            message_placeholder.error(response_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
        }
    )
