"""
app.py

Streamlit entrypoint for the LearnPath chatbot user interface

Key features:
- build_application(): wire AppService with GeminiClient, ChatMemory, SessionManager, messages
- Build and inject IntentDetector, ProfileExtractor and RoadmapService
- Manage st.session_state.application (AppService instance)
- Render header and chat interface, then persist state via app.to_session()
"""

import streamlit as st

from config import settings, Settings
from ai import GeminiClient, SYSTEM_PROMPT
from memory import ChatMemory
from config import DEFAULT_CONTEXT_MESSAGES, default_messages
from services import (
    AppService,
    ChatService,
    SessionManager,
    IntentDetector,
    ProfileExtractor,
    RoadmapService,
)

import ui.header
import ui.chat_display

def build_application(config: Settings | None = None) -> AppService:
    """
    Build AppService instance with configured LLM client, memory, session and messages

    Args:
        config: Optional Settings instance; defaults to global settings when None

    Returns:
        AppService wired with ChatService, SessionManager, ChatMemory and MessageProvider
    """
    if config is None:
        config = settings
    llm_client = GeminiClient(
        api_key=config.GEMINI_API_KEY,
        model_name=config.GEMINI_MODEL,
        request_timeout=60,
        stream_timeout=120,
        system_prompt=SYSTEM_PROMPT,
    )
    memory = ChatMemory()
    session = SessionManager(timeout_minutes=30)
    messages = default_messages
    chat_service = ChatService(llm_client=llm_client)
    roadmap_service = RoadmapService(
        llm_client=llm_client,
        max_retries=2
    )
    intent_detector = IntentDetector(llm_client=llm_client)
    profile_extractor = ProfileExtractor(llm_client=llm_client)
    
    return AppService(
        chat_service=chat_service,
        session_manager=session,
        messages=messages,
        memory=memory,
        intent_detector=intent_detector,
        profile_extractor=profile_extractor,
        roadmap_service=roadmap_service,
        chat_context_messages=DEFAULT_CONTEXT_MESSAGES,
    )

st.set_page_config(
    page_title="LearnPath Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed",
)

if "application" not in st.session_state:
    st.session_state.application = build_application()

app: AppService = st.session_state.application

ui.header.render_header(app)
ui.chat_display.render_chat_interface(app)
app.to_session(st.session_state)