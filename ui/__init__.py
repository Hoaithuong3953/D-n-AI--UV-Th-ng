"""
ui package

Presentation helpers for the LearnPath chatbot Streamlit UI

Key features:
- render_chat_interface: render chat history and stream events for new user input
- render_header: render page header with title, subtitle and reset-session button
"""
from .chat_display import render_chat_interface
from .header import render_header

__all__ = ["render_chat_interface", "render_header"]