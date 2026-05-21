# app.py
# Streamlit + Ollama chatbot with automatic image understanding support

import streamlit as st
import ollama

MODEL_OPTIONS = [
    "llama3",
    "gemma:2b",
    "gemma:latest",
]

VISION_MODEL_CANDIDATES = [
    "llava:latest",
    "llava",
    "llama3.2-vision:latest",
    "llama3.2-vision",
    "bakllava:latest",
    "bakllava",
    "moondream:latest",
    "moondream",
]

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
DEFAULT_IMAGE_PROMPT = "Please describe this image."


def list_local_models():
    try:
        response = ollama.list()
    except Exception:
        return []

    raw_models = getattr(response, "models", None)
    if raw_models is None and hasattr(response, "get"):
        raw_models = response.get("models", [])

    model_names = []
    for model in raw_models or []:
        if hasattr(model, "model"):
            model_names.append(model.model)
        elif isinstance(model, dict) and "model" in model:
            model_names.append(model["model"])

    return model_names


def is_vision_model(model_name):
    lowered = model_name.lower()
    return any(
        token in lowered
        for token in ("vision", "llava", "bakllava", "moondream")
    )


def pick_vision_model(selected_model, available_models):
    if is_vision_model(selected_model):
        return selected_model

    for candidate in VISION_MODEL_CANDIDATES:
        if candidate in available_models:
            return candidate

    for model_name in available_models:
        if is_vision_model(model_name):
            return model_name

    return VISION_MODEL_CANDIDATES[0]


def build_export_text(messages):
    lines = []

    for msg in messages:
        role = msg["role"].capitalize()
        content = (msg.get("content") or "").strip()
        image_name = msg.get("image_name")

        if image_name:
            lines.append(f"{role}: [Image attached: {image_name}]")

        if content:
            lines.append(f"{role}: {content}")

        lines.append("")

    return "\n".join(lines)


def build_model_messages(messages, active_image_index=None):
    model_messages = []

    for index, message in enumerate(messages):
        content = (message.get("content") or "").strip()
        image_name = message.get("image_name")

        if image_name:
            image_note = f"[Image attached: {image_name}]"
            content = f"{image_note}\n\n{content}" if content else image_note

        model_message = {
            "role": message["role"],
            "content": content,
        }

        if index == active_image_index and message.get("images"):
            model_message["images"] = message["images"]

        model_messages.append(model_message)

    return model_messages


def render_message(message):
    with st.chat_message(message["role"]):
        if message.get("image_name") and message.get("images"):
            st.image(message["images"][0], caption=message["image_name"])

        if message.get("content"):
            st.markdown(message["content"])


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="O",
    layout="centered",
)

# ---------------- SESSION STATE ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

available_models = list_local_models()

# ---------------- TITLE ---------------- #
st.title("Ollama Chatbot")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Settings")

MODEL = st.sidebar.selectbox(
    "Choose Model",
    MODEL_OPTIONS,
)

VISION_MODEL = pick_vision_model(MODEL, available_models)
st.sidebar.caption(
    f"Image requests will use `{VISION_MODEL}` when `{MODEL}` cannot read images."
)

# ---------------- CLEAR CHAT BUTTON ---------------- #
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.uploader_key += 1
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
    render_message(message)

# ---------------- IMAGE INPUT ---------------- #
uploaded_image = st.file_uploader(
    "Attach an image for your next message",
    type=IMAGE_TYPES,
    key=f"chat_image_{st.session_state.uploader_key}",
)

if uploaded_image is not None:
    st.image(uploaded_image, caption=uploaded_image.name)

send_image_only = uploaded_image is not None and st.button("Send Image")

# ---------------- USER INPUT ---------------- #
prompt = st.chat_input("Type your message...")

if prompt or send_image_only:
    user_prompt = prompt or DEFAULT_IMAGE_PROMPT
    image_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
    image_name = uploaded_image.name if uploaded_image is not None else None

    user_message = {
        "role": "user",
        "content": user_prompt,
    }

    if image_bytes is not None:
        user_message["images"] = [image_bytes]
        user_message["image_name"] = image_name

    st.session_state.messages.append(user_message)
    active_image_index = len(st.session_state.messages) - 1 if image_bytes is not None else None

    render_message(user_message)

    request_model = VISION_MODEL if image_bytes is not None else MODEL
    model_messages = build_model_messages(
        st.session_state.messages,
        active_image_index=active_image_index,
    )

    with st.chat_message("assistant"):
        if image_bytes is not None and request_model != MODEL:
            st.caption(f"Analyzing image with `{request_model}`")

        message_placeholder = st.empty()
        response_text = ""

        try:
            stream = ollama.chat(
                model=request_model,
                messages=model_messages,
                stream=True,
            )

            for chunk in stream:
                content = chunk["message"]["content"]
                response_text += content
                message_placeholder.markdown(response_text + "|")

            message_placeholder.markdown(response_text)

        except Exception as e:
            if image_bytes is not None:
                response_text = f"""
[Error] Error analyzing the image

Make sure:
- Ollama is installed
- Ollama is running
- The vision model `{request_model}` is pulled

Example:
`ollama pull {request_model}`

Technical Error:
{e}
""".strip()
            else:
                response_text = f"""
[Error] Error connecting to Ollama

Make sure:
- Ollama is installed
- Ollama is running
- The selected model `{request_model}` is pulled

Example:
`ollama pull {request_model}`

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

    st.session_state.uploader_key += 1
    st.rerun()
