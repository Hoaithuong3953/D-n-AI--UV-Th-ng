"""
chat_display.py

Streamlit UI for chat history and input; consumes AppService.handle_message() events

Key features:
- Render existing chat history from AppService memory
- On user input, stream events (TextChunk, StatusUpdate, ErrorOccurred, SessionExpired) and update UI
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import streamlit as st

from domain.events import (
    TextChunk,
    StatusUpdate,
    ErrorOccurred,
    SessionExpired,
)
from config import MessageKey

if TYPE_CHECKING:
    from services import AppService

def render_chat_interface(app: AppService) -> None:
    """
    Render chat history from AppService memory and stream events for new user input

    Args:
        app: AppService instance providing history, messages and handle_message()
    """
    history = app._memory.load_history()
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
            thinking_msg = app.messages.get(MessageKey.THINKING)
            placeholder.markdown(thinking_msg)
            full = ""
            showing_status = True

            for event in app.handle_message(user_input):
                match event:
                    case TextChunk(text=text):
                        if showing_status:
                            full = text
                            showing_status = False
                        else:
                            full += text
                        placeholder.markdown(full)
                    case StatusUpdate(message=message):
                        full = message
                        showing_status = True
                        placeholder.markdown(full)
                    case ErrorOccurred(user_message=user_message):
                        if showing_status:
                            full = user_message
                        else:
                            full += user_message
                        showing_status = False
                        placeholder.markdown(full)
                    case SessionExpired(message=message):
                        if showing_status:
                            full = message
                        else:
                            full += message
                        showing_status = False
                        placeholder.markdown(full)

            st.rerun()