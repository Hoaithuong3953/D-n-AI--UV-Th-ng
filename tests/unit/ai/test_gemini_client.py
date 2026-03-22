"""
test_gemini_client.py

Unit tests for GeminiClient (config validation, generate_text, stream_chat, _to_gemini_history)
"""
import pytest
from unittest.mock import MagicMock
from google.api_core import exceptions as google_exceptions

from ai.gemini_client import GeminiClient, _SAFETY_SETTINGS
from domain import ChatMessage
from utils import LLMServiceError, ValidationError

def test_validate_config_rejects_empty_api_key():
    """Constructor raises ValidationError when api_key is empty or whitespace"""
    with pytest.raises(ValidationError, match="Gemini API key must not be empty"):
        GeminiClient(
            api_key="",
            model_name="dummy-model",
            request_timeout=30,
            stream_timeout=30,
            system_prompt="dummy"
        )

def test_validate_config_rejects_empty_model_name():
    """Constructor raises ValidationError when model_name is empty or whitespace"""
    with pytest.raises(ValidationError, match="Gemini model name must not be empty"):
        GeminiClient(
            api_key="dummy-key",
            model_name="   ",
            request_timeout=30,
            stream_timeout=30,
            system_prompt="dummy-prompt"
        )

def test_validate_config_rejects_empty_system_prompt():
    """Constructor raises ValidationError when system_prompt is empty or whitespace"""
    with pytest.raises(ValidationError, match="System prompt must not be empty"):
        GeminiClient(
            api_key="dummy-key",
            model_name="dummy-model",
            request_timeout=30,
            stream_timeout=30,
            system_prompt="   "
        )

def test_init_model_invalid_argument_raises_validation_error(mock_genai_model):
    """GenerativeModel raises InvalidArgument -> ValidationError with config message"""
    model_cls, _, _ = mock_genai_model
    model_cls.side_effect = google_exceptions.InvalidArgument("bad request")

    with pytest.raises(ValidationError, match="Invalid Gemini API key or model name"):
        GeminiClient(
            api_key="dummy-key",
            model_name="dummy-model",
            request_timeout=30,
            stream_timeout=30,
            system_prompt="dummy-prompt",
        )

def test_init_model_unexpected_error_raises_llm_service_error(mock_genai_model):
    """Unexpected error during GenerativeModel init -> LLMServiceError LLM_INIT_FAILED"""
    model_cls, _, _ = mock_genai_model
    model_cls.side_effect = RuntimeError("sdk boom")

    with pytest.raises(LLMServiceError) as exc_info:
        GeminiClient(
            api_key="dummy-key",
            model_name="dummy-model",
            request_timeout=30,
            stream_timeout=30,
            system_prompt="dummy-prompt",
        )
    assert exc_info.value.code == "LLM_INIT_FAILED"

def test_init_model_configures_genai_and_uses_system_prompt(mock_genai_model):
    """genai.configure and GenerativeModel receive api_key, model_name and system_instruction"""
    model_cls, _, mock_configure = mock_genai_model

    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt"
    )

    mock_configure.assert_called_once_with(api_key="dummy-key")
    model_cls.assert_called_once()
    kwargs = model_cls.call_args.kwargs
    assert kwargs["model_name"] == "dummy-model"
    assert kwargs["system_instruction"] == "dummy-prompt"

def test_generate_text_success(mock_genai_model):
    """generate_text calls model.generate_content once and returns stripped response text"""
    _, model_instance, _ = mock_genai_model
    mock_response = MagicMock()
    mock_response.text = "Hello World"
    model_instance.generate_content.return_value = mock_response
    
    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt"
    )

    text = client.generate_text("prompt")

    model_instance.generate_content.assert_called_once()
    assert text == "Hello World"

    call_args = model_instance.generate_content.call_args
    assert call_args.kwargs["safety_settings"] == _SAFETY_SETTINGS
    assert call_args.kwargs["request_options"]["timeout"] == 30

def test_generate_text_google_api_error_raises_llm_service_error(mock_genai_model):
    """generate_text maps GoogleAPICallError from generate_content to LLMServiceError"""
    _, model_instance, _ = mock_genai_model
    model_instance.generate_content.side_effect = google_exceptions.InvalidArgument("bad")

    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt",
    )

    with pytest.raises(LLMServiceError) as exc_info:
        client.generate_text("prompt")
    assert exc_info.value.code == "GENERATION_FAILED"

def test_generate_text_strips_response_whitespace(mock_genai_model):
    """generate_text returns response.text stripped of leading/trailing whitespace"""
    _, model_instance, _ = mock_genai_model
    mock_response = MagicMock()
    mock_response.text = "  padded  "
    model_instance.generate_content.return_value = mock_response

    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt",
    )

    assert client.generate_text("prompt") == "padded"

def test_generate_text_empty_prompt_raises_validation_error(mock_genai_model):
    """generate_text raises ValidationError when prompt is empty or whitespace"""
    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt"
    )

    with pytest.raises(ValidationError, match="Prompt must not be empty"):
        client.generate_text("   ")

def test_generate_text_empty_response_raises_llm_service_error(mock_genai_model):
    """generate_text raises LLMServiceError when response has no usable text"""
    _, model_instance, _ = mock_genai_model
    mock_response = MagicMock()
    mock_response.text = ""
    model_instance.generate_content.return_value = mock_response

    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt"
    )

    with pytest.raises(LLMServiceError, match="Gemini returned empty response"):
        client.generate_text("prompt")

def test_generate_text_blocked_by_safety_raises_error(mock_genai_model):
    """generate_text raises LLMServiceError by safety settings"""
    _, model_instance, _ = mock_genai_model
    mock_response = MagicMock()
    mock_response.candidates = []
    mock_response.text = None
    model_instance.generate_content.return_value = mock_response

    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt"
    )

    with pytest.raises(LLMServiceError):
        client.generate_text("dummy-prompt")

def test_to_gemini_history_map_roles_correctly():
    """_to_gemini_history maps ChatMessage roles to Gemini format with parts from content"""
    history = [
        ChatMessage(role="user", content="user msg"),
        ChatMessage(role="assistant", content="assistant msg"),
        ChatMessage(role="system", content="system msg"),
    ]

    messages = GeminiClient._to_gemini_history(history)

    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["parts"] == ["user msg"]
    assert messages[1]["role"] == "model"
    assert messages[1]["parts"] == ["assistant msg"]
    assert messages[2]["role"] == "user"
    assert messages[2]["parts"] == ["system msg"]

def test_stream_chat_happy_path(mock_genai_model):
    """stream_chat calls start_chat and send_message(stream=True) and yields chunk texts"""
    _, model_instance, _ = mock_genai_model

    def fake_stream():
        yield MagicMock(text="Chunk1")
        yield MagicMock(text="Chunk2")

    fake_chat = MagicMock()
    fake_chat.send_message.return_value = fake_stream()

    model_instance.start_chat.return_value = fake_chat

    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt"
    )

    history = [ChatMessage(role="user", content="msg1")]
    chunks = list(client.stream_chat(history=history, new_message="new_msg"))

    assert chunks == ["Chunk1", "Chunk2"]
    model_instance.start_chat.assert_called_once()
    fake_chat.send_message.assert_called_once()

    call_kwargs = fake_chat.send_message.call_args.kwargs
    assert call_kwargs["safety_settings"] == _SAFETY_SETTINGS
    assert call_kwargs["request_options"]["timeout"] == 30
    assert call_kwargs["stream"] is True

def test_stream_chat_google_api_error_raises_llm_service_error(mock_genai_model):
    """stream_chat maps GoogleAPICallError from send_message to LLMServiceError"""
    _, model_instance, _ = mock_genai_model
    fake_chat = MagicMock()
    fake_chat.send_message.side_effect = google_exceptions.InvalidArgument("bad")
    model_instance.start_chat.return_value = fake_chat

    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt",
    )

    with pytest.raises(LLMServiceError) as exc_info:
        list(client.stream_chat(history=[], new_message="hi"))
    assert exc_info.value.code == "STREAM_FAILED"

def test_stream_chat_skips_chunks_without_text(mock_genai_model):
    """stream_chat yields only chunks where chunk.text is truthy"""
    _, model_instance, _ = mock_genai_model

    def fake_stream():
        yield MagicMock(text=None)
        yield MagicMock(text="")
        yield MagicMock(text="ok")

    fake_chat = MagicMock()
    fake_chat.send_message.return_value = fake_stream()
    model_instance.start_chat.return_value = fake_chat

    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt",
    )

    assert list(client.stream_chat(history=[], new_message="x")) == ["ok"]

def test_stream_chat_empty_new_message_raises_validation_error(mock_genai_model):
    """stream_chat raises ValidationError when new_message is empty or whitespace"""
    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt"
    )

    with pytest.raises(ValidationError, match="New message must be not empty"):
        list(client.stream_chat(history=[], new_message=" "))

def test_safety_settings_structure_is_correct():
    """_SAFETY_SETTINGS has correct structure and values"""
    expected_categories = {
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    }
    assert isinstance(_SAFETY_SETTINGS, list)
    assert len(_SAFETY_SETTINGS) == 4

    for setting in _SAFETY_SETTINGS:
        assert isinstance(setting, dict)
        assert "category" in setting
        assert "threshold" in setting
        assert setting["category"] in expected_categories
        assert setting["threshold"] == "BLOCK_ONLY_HIGH"