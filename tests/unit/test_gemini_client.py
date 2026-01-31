
import pytest
from unittest.mock import MagicMock

from ai import GeminiClient
from domain import ChatMessage
from utils import LLMServiceError, ValidationError

def test_validate_config_rejects_empty_api_key():
    """Constructor should raise ValidationError when api_key is empty or whitespace"""
    with pytest.raises(ValidationError):
        GeminiClient(
            api_key="",
            model_name="dummy-model",
            request_timeout=30,
            stream_timeout=30,
            system_prompt="dummy"
        )

def test_validate_config_rejects_empty_model_name():
    """Constructor should raise ValidationError when model_name is empty or whitespace"""
    with pytest.raises(ValidationError):
        GeminiClient(
            api_key="dummy-key",
            model_name="   ",
            request_timeout=30,
            stream_timeout=30,
            system_prompt="dummy-prompt"
        )

def test_validate_config_rejects_empty_system_prompt():
    """Constructor should raise ValidationError when system_prompt is empty or whitespace"""
    with pytest.raises(ValidationError):
        GeminiClient(
            api_key="dummy-key",
            model_name="dummy-model",
            request_timeout=30,
            stream_timeout=30,
            system_prompt="   "
        )

def test_init_model_configures_genai_and_uses_system_prompt(mock_genai_model):
    """genai.configure and GenerativeModel should receive api_key, model_name and system_instruction"""
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
    """generate_text should call model.generate_content once and return stripped response text"""
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

def test_generate_text_empty_prompt_raises_validation_error(mock_genai_model):
    """generate_text should raise ValidationError when prompt is empty or whitespace"""
    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt"
    )

    with pytest.raises(ValidationError):
        client.generate_text("   ")

def test_generate_text_empty_response_raises_llm_service_error(mock_genai_model):
    """generate_text should raise LLMServiceError when response has no usable text"""
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

    with pytest.raises(LLMServiceError):
        client.generate_text("prompt")

def test_to_gemini_history_map_roles_correctly():
    """_to_gemini_history should map ChatMessage roles to Gemini format with parts from content"""
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
    """stream_chat should call start_chat and send_message(stream=True) and yield chunk texts"""
    _, model_instance, _ = mock_genai_model

    def fake_stream(*args, **kwargs):
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

def test_stream_chat_empty_new_message_raises_validation_error(mock_genai_model):
    """stream_chat should raise ValidationError when new_message is empty or whitespace"""
    client = GeminiClient(
        api_key="dummy-key",
        model_name="dummy-model",
        request_timeout=30,
        stream_timeout=30,
        system_prompt="dummy-prompt"
    )

    with pytest.raises(ValidationError):
        list(client.stream_chat(history=[], new_message=" "))