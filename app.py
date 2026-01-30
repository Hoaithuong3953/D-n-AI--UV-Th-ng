"""
Streamlit entrypoint for the LearnPath chatbot user interface
"""

import streamlit as st

from config import settings
from ai import GeminiClient, SYSTEM_PROMPT
from memory import ChatMemory
from services import ChatService

def build_chat_service() -> ChatService:
    ai_client = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model_name=settings.GEMINI_MODEL,
        request_timeout=60,
        stream_timeout=120,
        system_prompt=SYSTEM_PROMPT
    )
    history = ChatMemory(storage=[])
    return ChatService(ai_client=ai_client, history=history)

st.set_page_config(page_title="LearnPath Chatbot", layout="centered")
st.title("LearnPath Chatbot")
st.markdown("Trợ lý AI tạo lộ trình học tập cá nhân hóa")

if "chat_service" not in st.session_state:
    st.session_state.chat_service = build_chat_service()

history = st.session_state.chat_service.history.load_history()
for msg in history:
    role = "assistant" if msg.role == "assistant" else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Nhập tin nhắn...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        for chunk in st.session_state.chat_service.process_message(user_input):
            full += chunk
            placeholder.markdown(full)