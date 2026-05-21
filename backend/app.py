# app.py
# Streamlit + Ollama chatbot with email login and user chat history

import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
import ssl
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from uuid import uuid4

import ollama
import streamlit as st

MODEL_OPTIONS = [
    "llama3",
    "gemma:2b",
    "gemma:latest",
]

AUTH_VIEWS = [
    "Sign In",
    "Create Account",
    "Verify Email",
]

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PASSWORD_MIN_LENGTH = 8
VERIFICATION_CODE_LENGTH = 6
VERIFICATION_CODE_TTL_MINUTES = 10
UNSAVED_CHAT_OPTION = "__current__"

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "chat_history.json"
USERS_FILE = BASE_DIR / "users.json"
MAIL_SETTINGS_FILE = BASE_DIR / "mail_settings.json"


def load_json_list(path):
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    return data if isinstance(data, list) else []


def save_json_list(path, items):
    path.write_text(
        json.dumps(items, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def load_json_object(path):
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def normalize_email(email):
    return (email or "").strip().lower()


def is_valid_email(email):
    return bool(EMAIL_PATTERN.fullmatch(normalize_email(email)))


def hash_secret(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120000,
    ).hex()
    return f"{salt}${digest}"


def verify_password(password, stored_hash):
    if "$" not in stored_hash:
        return False

    salt, expected_digest = stored_hash.split("$", 1)
    candidate_digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120000,
    ).hex()
    return hmac.compare_digest(candidate_digest, expected_digest)


def generate_verification_code():
    upper_bound = 10 ** VERIFICATION_CODE_LENGTH
    return f"{secrets.randbelow(upper_bound):0{VERIFICATION_CODE_LENGTH}d}"


def get_config_value(name, default=""):
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        env_value = os.getenv(name, "").strip()
        if env_value:
            return env_value

    saved_settings = load_json_object(MAIL_SETTINGS_FILE)
    saved_value = str(saved_settings.get(name, "")).strip()
    return saved_value or default


def get_mail_config():
    host = get_config_value("SMTP_HOST")
    port_text = get_config_value("SMTP_PORT", "587")
    username = get_config_value("SMTP_USERNAME")
    password = get_config_value("SMTP_PASSWORD")
    from_email = get_config_value("SMTP_FROM_EMAIL") or username

    if not host:
        raise ValueError("Missing SMTP_HOST.")

    if not port_text.isdigit():
        raise ValueError("SMTP_PORT must be a number.")

    if not username or not password:
        raise ValueError("Missing SMTP_USERNAME or SMTP_PASSWORD.")

    if not from_email:
        raise ValueError("Missing SMTP_FROM_EMAIL.")

    return {
        "host": host,
        "port": int(port_text),
        "username": username,
        "password": password,
        "from_email": from_email,
    }


def mail_setup_hint():
    return (
        "Set the SMTP values in `backend/mail_settings.json`. "
        "For Gmail, use SMTP host smtp.gmail.com, port 587, your Gmail address, "
        "and a Google app password."
    )


def set_auth_notice(kind, text):
    st.session_state.auth_notice = {
        "kind": kind,
        "text": text,
    }


def send_verification_email(recipient_email, code):
    config = get_mail_config()

    message = EmailMessage()
    message["Subject"] = "Your Ollama Chatbot verification code"
    message["From"] = config["from_email"]
    message["To"] = recipient_email
    message.set_content(
        "\n".join(
            [
                "Your Ollama Chatbot verification code is:",
                "",
                code,
                "",
                f"This code expires in {VERIFICATION_CODE_TTL_MINUTES} minutes.",
            ]
        )
    )

    context = ssl.create_default_context()

    with smtplib.SMTP(config["host"], config["port"], timeout=20) as server:
        server.starttls(context=context)
        server.login(config["username"], config["password"])
        server.send_message(message)


def load_users():
    users = []

    for user in load_json_list(USERS_FILE):
        if not isinstance(user, dict):
            continue

        email = normalize_email(user.get("email"))
        password_hash = str(user.get("password_hash") or "")

        if not email or not password_hash:
            continue

        users.append(
            {
                "email": email,
                "password_hash": password_hash,
                "verified": bool(user.get("verified", False)),
                "verification_code_hash": str(user.get("verification_code_hash") or ""),
                "verification_expires_at": str(user.get("verification_expires_at") or ""),
                "created_at": str(user.get("created_at") or ""),
                "last_login_at": str(user.get("last_login_at") or ""),
            }
        )

    return users


def save_users(users):
    save_json_list(USERS_FILE, users)


def find_user(users, email):
    normalized_email = normalize_email(email)

    for index, user in enumerate(users):
        if normalize_email(user.get("email")) == normalized_email:
            return index, user

    return None, None


def load_all_chats():
    chats = []

    for chat in load_json_list(HISTORY_FILE):
        if not isinstance(chat, dict):
            continue

        chat_id = str(chat.get("id") or "").strip()
        messages = chat.get("messages")

        if not chat_id or not isinstance(messages, list):
            continue

        chats.append(
            {
                "id": chat_id,
                "title": str(chat.get("title") or "Untitled Chat"),
                "model": str(chat.get("model") or MODEL_OPTIONS[0]),
                "messages": messages,
                "updated_at": str(chat.get("updated_at") or ""),
                "owner_email": normalize_email(chat.get("owner_email")),
            }
        )

    return chats


def save_all_chats(chats):
    save_json_list(HISTORY_FILE, chats)


def migrate_legacy_chats(owner_email):
    all_chats = load_all_chats()

    if not any(not chat["owner_email"] for chat in all_chats):
        return

    verified_users = [
        user for user in load_users() if user.get("verified")
    ]

    if len(verified_users) != 1:
        return

    if normalize_email(verified_users[0].get("email")) != normalize_email(owner_email):
        return

    updated = False

    for chat in all_chats:
        if not chat["owner_email"]:
            chat["owner_email"] = normalize_email(owner_email)
            updated = True

    if updated:
        save_all_chats(all_chats)


def load_chat_history(owner_email):
    normalized_email = normalize_email(owner_email)
    chats = [
        chat for chat in load_all_chats()
        if chat["owner_email"] == normalized_email
    ]
    chats.sort(key=lambda chat: chat["updated_at"], reverse=True)
    return chats


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


def upsert_chat(owner_email, chat_id, model, messages):
    if not messages:
        return

    all_chats = load_all_chats()
    normalized_email = normalize_email(owner_email)
    now = datetime.now().isoformat(timespec="seconds")
    new_chat = {
        "id": chat_id,
        "title": build_chat_title(messages),
        "model": model,
        "messages": messages,
        "updated_at": now,
        "owner_email": normalized_email,
    }

    updated = False

    for index, chat in enumerate(all_chats):
        if chat["id"] == chat_id and chat["owner_email"] == normalized_email:
            all_chats[index] = new_chat
            updated = True
            break

    if not updated:
        all_chats.append(new_chat)

    all_chats.sort(key=lambda chat: chat["updated_at"], reverse=True)
    save_all_chats(all_chats)


def delete_chat(owner_email, chat_id):
    normalized_email = normalize_email(owner_email)
    chats = [
        chat for chat in load_all_chats()
        if not (chat["id"] == chat_id and chat["owner_email"] == normalized_email)
    ]
    save_all_chats(chats)


def build_export_text(messages):
    lines = []

    for msg in messages:
        role = msg["role"].capitalize()
        content = (msg.get("content") or "").strip()

        if content:
            lines.append(f"{role}: {content}")
            lines.append("")

    return "\n".join(lines)


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


def resolve_default_model_for_user(owner_email):
    history_entries = load_chat_history(owner_email)

    if history_entries and history_entries[0]["model"] in MODEL_OPTIONS:
        return history_entries[0]["model"]

    return MODEL_OPTIONS[0]


def login_user(email):
    normalized_email = normalize_email(email)
    migrate_legacy_chats(normalized_email)
    st.session_state.authenticated = True
    st.session_state.user_email = normalized_email
    st.session_state.selected_model = resolve_default_model_for_user(normalized_email)
    st.session_state.pending_verification_email = normalized_email
    start_new_chat()


def logout_user():
    st.session_state.authenticated = False
    st.session_state.user_email = ""
    st.session_state.messages = []
    st.session_state.current_chat_id = str(uuid4())
    st.session_state.selected_model = MODEL_OPTIONS[0]
    st.session_state.pending_verification_email = ""
    st.session_state.auth_view = AUTH_VIEWS[0]


def handle_sign_in(email, password):
    normalized_email = normalize_email(email)

    if not normalized_email or not password:
        st.error("Enter both email and password.")
        return

    users = load_users()
    user_index, user = find_user(users, normalized_email)

    if user is None:
        st.error("No account exists for that email.")
        return

    if not user.get("verified"):
        st.session_state.pending_verification_email = normalized_email
        st.session_state.auth_view = "Verify Email"
        set_auth_notice(
            "info",
            "That account is not verified yet. Enter the code from your email.",
        )
        st.rerun()

    if not verify_password(password, user.get("password_hash", "")):
        st.error("Incorrect password.")
        return

    users[user_index]["last_login_at"] = datetime.now().isoformat(timespec="seconds")
    save_users(users)
    login_user(normalized_email)
    st.rerun()


def handle_create_account(email, password, confirm_password):
    normalized_email = normalize_email(email)

    if not is_valid_email(normalized_email):
        st.error("Enter a real email address.")
        return

    if len(password) < PASSWORD_MIN_LENGTH:
        st.error(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.")
        return

    if password != confirm_password:
        st.error("Passwords do not match.")
        return

    verification_code = generate_verification_code()
    verification_expires_at = (
        datetime.now() + timedelta(minutes=VERIFICATION_CODE_TTL_MINUTES)
    ).isoformat(timespec="seconds")

    try:
        send_verification_email(normalized_email, verification_code)
    except Exception as exc:
        st.error(f"Could not send verification email. {exc}")
        st.info(mail_setup_hint())
        return

    users = load_users()
    user_index, existing_user = find_user(users, normalized_email)
    user_record = {
        "email": normalized_email,
        "password_hash": hash_password(password),
        "verified": False,
        "verification_code_hash": hash_secret(verification_code),
        "verification_expires_at": verification_expires_at,
        "created_at": (
            existing_user.get("created_at")
            if existing_user is not None
            else datetime.now().isoformat(timespec="seconds")
        ),
        "last_login_at": (
            existing_user.get("last_login_at")
            if existing_user is not None
            else ""
        ),
    }

    if existing_user and existing_user.get("verified"):
        st.error("An account with that email already exists. Please sign in.")
        return

    if user_index is None:
        users.append(user_record)
    else:
        users[user_index] = user_record

    save_users(users)
    st.session_state.pending_verification_email = normalized_email
    st.session_state.auth_view = "Verify Email"
    set_auth_notice(
        "success",
        "Verification code sent. Check your inbox and verify your email.",
    )
    st.rerun()


def handle_verify_email(email, code):
    normalized_email = normalize_email(email)
    cleaned_code = (code or "").strip()

    if not is_valid_email(normalized_email):
        st.error("Enter the same email address you used to register.")
        return

    if not cleaned_code:
        st.error("Enter the verification code from your email.")
        return

    users = load_users()
    user_index, user = find_user(users, normalized_email)

    if user is None:
        st.error("No pending account was found for that email.")
        return

    if user.get("verified"):
        login_user(normalized_email)
        st.success("Email already verified. You are signed in now.")
        st.rerun()
        return

    expires_at = user.get("verification_expires_at", "")

    try:
        expiry = datetime.fromisoformat(expires_at)
    except ValueError:
        expiry = datetime.min

    if datetime.now() > expiry:
        st.error("That verification code has expired. Request a new one.")
        return

    if not hmac.compare_digest(
        hash_secret(cleaned_code),
        user.get("verification_code_hash", ""),
    ):
        st.error("Verification code is incorrect.")
        return

    users[user_index]["verified"] = True
    users[user_index]["verification_code_hash"] = ""
    users[user_index]["verification_expires_at"] = ""
    save_users(users)
    login_user(normalized_email)
    st.success("Email verified. You are signed in now.")
    st.rerun()


def handle_resend_code(email):
    normalized_email = normalize_email(email)

    if not is_valid_email(normalized_email):
        st.error("Enter a valid email address first.")
        return

    users = load_users()
    user_index, user = find_user(users, normalized_email)

    if user is None:
        st.error("No account exists for that email.")
        return

    if user.get("verified"):
        st.info("That email is already verified. You can sign in now.")
        return

    verification_code = generate_verification_code()
    verification_expires_at = (
        datetime.now() + timedelta(minutes=VERIFICATION_CODE_TTL_MINUTES)
    ).isoformat(timespec="seconds")

    try:
        send_verification_email(normalized_email, verification_code)
    except Exception as exc:
        st.error(f"Could not send verification email. {exc}")
        st.info(mail_setup_hint())
        return

    users[user_index]["verification_code_hash"] = hash_secret(verification_code)
    users[user_index]["verification_expires_at"] = verification_expires_at
    save_users(users)
    st.session_state.pending_verification_email = normalized_email
    st.success("A new verification code has been sent.")


def render_auth_page():
    st.title("Ollama Chatbot")
    st.subheader("Email Login")
    st.caption("Use a real email address. New accounts are verified with a real email code.")

    notice = st.session_state.pop("auth_notice", None)
    if isinstance(notice, dict):
        notice_kind = notice.get("kind")
        notice_text = notice.get("text", "")

        if notice_kind == "success":
            st.success(notice_text)
        elif notice_kind == "info":
            st.info(notice_text)
        else:
            st.warning(notice_text)

    selected_view = st.radio(
        "Account",
        AUTH_VIEWS,
        index=AUTH_VIEWS.index(st.session_state.auth_view),
        horizontal=True,
    )

    if selected_view != st.session_state.auth_view:
        st.session_state.auth_view = selected_view

    active_view = selected_view

    if active_view == "Sign In":
        with st.form("sign_in_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            handle_sign_in(email, password)

    elif active_view == "Create Account":
        st.info(mail_setup_hint())

        with st.form("create_account_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)

        if submitted:
            handle_create_account(email, password, confirm_password)

    else:
        st.info(mail_setup_hint())

        default_email = st.session_state.pending_verification_email

        with st.form("verify_email_form"):
            email = st.text_input("Email", value=default_email)
            code = st.text_input("Verification Code")
            verify_submitted = st.form_submit_button("Verify Email", use_container_width=True)
            resend_submitted = st.form_submit_button("Resend Code", use_container_width=True)

        if verify_submitted:
            handle_verify_email(email, code)

        if resend_submitted:
            handle_resend_code(email)


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="O",
    layout="centered",
)

# ---------------- SESSION STATE ---------------- #
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid4())

if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_OPTIONS[0]

if "pending_verification_email" not in st.session_state:
    st.session_state.pending_verification_email = ""

if "auth_view" not in st.session_state:
    st.session_state.auth_view = AUTH_VIEWS[0]

if not st.session_state.authenticated:
    render_auth_page()
    st.stop()

if st.session_state.selected_model not in MODEL_OPTIONS:
    st.session_state.selected_model = MODEL_OPTIONS[0]

history_entries = load_chat_history(st.session_state.user_email)
history_lookup = {chat["id"]: chat for chat in history_entries}

# ---------------- TITLE ---------------- #
st.title("Ollama Chatbot")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Account")
st.sidebar.caption(f"Signed in as `{st.session_state.user_email}`")

if st.sidebar.button("Log Out"):
    logout_user()
    st.rerun()

st.sidebar.header("Settings")

if st.sidebar.button("New Chat"):
    start_new_chat()
    st.rerun()

if (
    st.session_state.current_chat_id in history_lookup
    and st.sidebar.button("Delete This Chat")
):
    delete_chat(st.session_state.user_email, st.session_state.current_chat_id)
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
    st.sidebar.caption("No saved chats yet for this account.")

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
        owner_email=st.session_state.user_email,
        chat_id=st.session_state.current_chat_id,
        model=MODEL,
        messages=st.session_state.messages,
    )
