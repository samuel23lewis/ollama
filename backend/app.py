# app.py
# Streamlit + Ollama chatbot with persistent chat history

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import ollama
import streamlit as st

MODEL_OPTIONS = [
    "llama3",
    "gemma:2b",
    "gemma:latest",
]

HISTORY_FILE = Path(__file__).with_name("chat_history.json")
UNSAVED_CHAT_OPTION = "__current__"


def build_export_text(messages):
    lines = []

    for msg in messages:
        role = msg["role"].capitalize()
        content = (msg.get("content") or "").strip()

        if content:
            lines.append(f"{role}: {content}")
            lines.append("")

    return "\n".join(lines)


def load_chat_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        raw_data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(raw_data, list):
        return []

    chats = []

    for item in raw_data:
        if not isinstance(item, dict):
            continue

        chat_id = str(item.get("id") or "").strip()
        messages = item.get("messages")

        if not chat_id or not isinstance(messages, list):
            continue

        chats.append(
            {
                "id": chat_id,
                "title": str(item.get("title") or "Untitled Chat"),
                "model": str(item.get("model") or MODEL_OPTIONS[0]),
                "messages": messages,
                "updated_at": str(item.get("updated_at") or ""),
            }
        )

    chats.sort(key=lambda chat: chat["updated_at"], reverse=True)
    return chats


def save_chat_history(chats):
    HISTORY_FILE.write_text(
        json.dumps(chats, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def build_chat_title(messages):
    for message in messages:
        if message.get("role") != "user":
            continue

        content = (message.get("content") or "").strip()
        if content:
            return content[:40] + ("..." if len(content) > 40 else "")

    return "Untitled Chat"


def format_timestamp(timestamp):
    if not timestamp:
        return "Unknown time"

    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp

    return parsed.strftime("%Y-%m-%d %H:%M")


def upsert_chat(chat_id, model, messages):
    if not messages:
        return

    chats = load_chat_history()
    now = datetime.now().isoformat(timespec="seconds")
    new_chat = {
        "id": chat_id,
        "title": build_chat_title(messages),
        "model": model,
        "messages": messages,
        "updated_at": now,
    }

    updated = False

    for index, chat in enumerate(chats):
        if chat["id"] == chat_id:
            chats[index] = new_chat
            updated = True
            break

    if not updated:
        chats.append(new_chat)

    chats.sort(key=lambda chat: chat["updated_at"], reverse=True)
    save_chat_history(chats)


def delete_chat(chat_id):
    chats = [chat for chat in load_chat_history() if chat["id"] != chat_id]
    save_chat_history(chats)


def start_new_chat():
    st.session_state.messages = []
    st.session_state.current_chat_id = str(uuid4())


def load_saved_chat(chat):
    st.session_state.messages = chat["messages"]
    st.session_state.current_chat_id = chat["id"]

    if chat["model"] in MODEL_OPTIONS:
        st.session_state.selected_model = chat["model"]


def format_history_label(option, history_lookup):
    if option == UNSAVED_CHAT_OPTION:
        return "Current Chat (unsaved)"

    chat = history_lookup[option]
    return (
        f"{chat['title']} | {chat['model']} | "
        f"{format_timestamp(chat['updated_at'])}"
    )


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="O",
    layout="centered",
)

# ---------------- SESSION STATE ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid4())

if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_OPTIONS[0]

history_entries = load_chat_history()
history_lookup = {chat["id"]: chat for chat in history_entries}

# ---------------- TITLE ---------------- #
st.title("Ollama Chatbot")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Settings")

if st.sidebar.button("New Chat"):
    start_new_chat()
    st.rerun()

if (
    st.session_state.current_chat_id in history_lookup
    and st.sidebar.button("Delete This Chat")
):
    delete_chat(st.session_state.current_chat_id)
    start_new_chat()
    st.rerun()

st.sidebar.download_button(
    label="Export Chat",
    data=build_export_text(st.session_state.messages),
    file_name="chat_history.txt",
    mime="text/plain",
)

st.sidebar.subheader("Chat History")

if history_entries:
    history_options = [chat["id"] for chat in history_entries]

    if st.session_state.current_chat_id not in history_lookup:
        history_options = [UNSAVED_CHAT_OPTION] + history_options
        current_option = UNSAVED_CHAT_OPTION
    else:
        current_option = st.session_state.current_chat_id

    selected_history_option = st.sidebar.radio(
        "Saved conversations",
        options=history_options,
        index=history_options.index(current_option),
        format_func=lambda option: format_history_label(option, history_lookup),
    )

    if (
        selected_history_option != UNSAVED_CHAT_OPTION
        and selected_history_option != st.session_state.current_chat_id
    ):
        load_saved_chat(history_lookup[selected_history_option])
        st.rerun()
else:
    st.sidebar.caption("No saved chats yet.")

MODEL = st.sidebar.selectbox(
    "Choose Model",
    MODEL_OPTIONS,
    key="selected_model",
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

    upsert_chat(
        chat_id=st.session_state.current_chat_id,
        model=MODEL,
        messages=st.session_state.messages,
    )
