"""
test_chat_flow_happy_path.py

Integration tests for successful chat flow scenarios

Tests:
- Event stream generation (StatusUpdate -> TextChunks)
- Chat history persistence
- Multiple message exchanges
- Session activity tracking
"""
import pytest
import time

from domain.events import TextChunk, StatusUpdate

class TestChatFlowHappyPath:
    """Integration tests for successful chat flow scenarios"""

    def test_user_message_generates_correct_event_stream(self, app_service,fake_llm_client):
        """User sends message -> receives StatusUpdate, TextChunks in correct order"""
        user_input = "What is Python?"
        llm = fake_llm_client(
            ["Python ", "is ", "a powerful ", "programming ", "language"]
        )
        app = app_service(llm_client=llm)
        events = list(app.handle_message(user_input))

        assert len(events) >= 2

        assert isinstance(events[0], StatusUpdate)
        assert events[0].status == "loading"

        text_chunks = [e for e in events[1:] if isinstance(e, TextChunk)]
        assert len(text_chunks) == 5

        full_response = "".join(chunk.text for chunk in text_chunks)
        assert full_response == "Python is a powerful programming language"

    def test_chat_history_persisted_correctly(self, app_service, fake_llm_client):
        """After chat, both user message and assistant response are saved"""
        user_input = "What is Python?"
        llm = fake_llm_client(
            ["Python ", "is ", "a powerful ", "programming ", "language"]
        )
        app = app_service(llm_client=llm)
        
        list(app.handle_message(user_input))

        history = app._memory.load_history()
        assert len(history) == 2

        assert history[0].role == "user"
        assert history[0].content == "What is Python?"

        assert history[1].role == "assistant"
        assert history[1].content == "Python is a powerful programming language"

    def test_multiple_messages_build_history(self, app_service, fake_llm_client):
        """Multiple exchanges build up conversation history"""
        llm = fake_llm_client(
            [
                "Python ", "is ", "a powerful ", "programming ", "language",
                "Depending on ", "the ", "level ", "and ", "duration"
            ]
        )
        app = app_service(llm_client=llm)

        list(app.handle_message("What is Python?"))
        list(app.handle_message("How long does it take to learn Python?"))

        history = app._memory.load_history()
        assert len(history) == 4
        assert history[0].content == "What is Python?"
        assert history[2].content == "How long does it take to learn Python?"

    def test_session_activity_updated(self, app_service, fake_llm_client):
        """Session activity timestamp updated after message"""
        llm = fake_llm_client(["Hello ", "World"])
        app = app_service(llm_client=llm)

        list(app.handle_message("Hello"))
        initial_activity = app._session.get_last_activity()

        time.sleep(0.01)

        list(app.handle_message("World"))

        new_activity = app._session.get_last_activity()
        assert new_activity > initial_activity