"""
roadmap_flow.py

Roadmap and profile-related flows for AppService

Key features:
- handle_roadmap_request: extract profile if needed, then generate roadmap
- handle_profile_reextraction: re-extract profile after user provides missing info
- generate_roadmap: generate roadmap from current profile and emit events
"""
from __future__ import annotations

from typing import Generator, TYPE_CHECKING

from domain import ChatMessage, ConversationState
from domain.snapshots import from_user_profile, from_roadmap
from domain.events import (
    Event,
    TextChunk,
    StatusUpdate,
    ProfileExtracted,
    RoadmapCreated,
    ErrorOccurred,
)
from config.messages import MessageKey
from utils import LLMServiceError, ValidationError, logger
from services.support.profile_inference import infer_duration_weeks_from_user_text

if TYPE_CHECKING:
    from services.app_service import AppService

def _missing_field_prompt(app: AppService, missing: list[str]) -> str:
    """Build missing-profile guidance message from missing required fields"""
    if not missing:
        return app.messages.get(MessageKey.FILL_PROFILE)
    
    field_to_message_key = {
        "goal": MessageKey.FILL_PROFILE_MISSING_GOAL,
        "level": MessageKey.FILL_PROFILE_MISSING_LEVEL,
        "time": MessageKey.FILL_PROFILE_MISSING_TIME,
    }

    numbered = []
    for idx, field in enumerate(missing, start=1):
        key = field_to_message_key.get(field)
        if key is None:
            label = app.messages.get(MessageKey.FILL_PROFILE).split("\n", 1)[0].strip()
        else:
            label = app.messages.get(key).split("\n\n", 1)[0].strip()
        numbered.append(f"{idx}. {label}")

    return app.messages.format(
        MessageKey.FILL_PROFILE_MISSING_FIELDS,
        missing_msg="\n".join(numbered),
    )

def handle_roadmap_request(app: AppService) -> Generator[Event, None, None]:
    """
    Handle roadmap flow: extract profile if needed, generate roadmap
        
    Flow:
    1. Check if profile exists
    2. If not, extract from history (yield ProfileExtracted or ErrorOccurred)
    3. Generate roadmap (yield RoadmapCreated or ErrorOccurred)

    Yields:
        StatusUpdate: Analyzing profile, generating roadmap
        ProfileExtracted: Profile successfully extracted
        TextChunk: Profile confirmation message
        RoadmapCreated: Roadmap successfully generated
        ErrorOccurred: Missing profile fields, LLM error, validation error
    """
    logger.info("roadmap request start")
    profile = app.current_profile
    if not profile:
        logger.info("extracting profile")
        yield StatusUpdate("analyzing_profile", app.messages.get(MessageKey.PROFILE_ANALYZING))
        try:
            history = app._get_recent_history()
            profile = app._profile_extractor.extract(history)
        except LLMServiceError:
            msg = app.messages.get(MessageKey.LLM_ERROR)
            yield ErrorOccurred("llm", msg)
            app._memory.add_message(ChatMessage(role="assistant", content=msg))
            return
        if profile:
            app.current_profile = profile
            yield ProfileExtracted(from_user_profile(profile))
            yield TextChunk(
                app.messages.format(
                    MessageKey.PROFILE_EXTRACTED,
                    goal=profile.goal,
                    level=profile.current_level,
                    time=profile.time_commitment,
                )
                + "\n\n"
            )
    
    if not profile:
        missing = getattr(app._profile_extractor, "last_missing_fields", [])
        fill_msg = _missing_field_prompt(app, missing)

        yield ErrorOccurred("validation", fill_msg)
        app._memory.add_message(ChatMessage(role="assistant", content=fill_msg))

        app.conversation_state = ConversationState.AWAITING_PROFILE_INFO
        logger.info("State changed to AWAITING_PROFILE_INFO")
        return
    
    yield from generate_roadmap(app)

def handle_profile_reextraction(app: AppService) -> Generator[Event, None, None]:
    """
    Re-extract profile after user provides missing information

    Args:
        app: AppService instance with memory, profile extractor and message provider

    Yields:
        StatusUpdate: Profile analyzing state
        ProfileExtracted: Profile extracted successfully
        TextChunk: Profile confirmation or still-incomplete guidance message
        RoadmapCreated: Generated roadmap after successful extraction
        ErrorOccurred: LLM or unexpected error during re-extraction
    """
    logger.info("Attempting profile re-extraction")
    yield StatusUpdate("analyzing_profile", app.messages.get(MessageKey.PROFILE_ANALYZING))

    try:
        history = app._get_recent_history()
        profile = app._profile_extractor.extract(history)
    except LLMServiceError:
        app.conversation_state = ConversationState.NORMAL
        msg = app.messages.get(MessageKey.LLM_ERROR)
        yield ErrorOccurred("llm", msg)
        app._memory.add_message(ChatMessage(role="assistant", content=msg))
        return
    except Exception as e:
        app.conversation_state = ConversationState.NORMAL
        logger.exception(f"Profile re-extraction failed: {e}")
        msg = app.messages.get(MessageKey.UNEXPECTED_ERROR)
        yield ErrorOccurred("unexpected", msg)
        app._memory.add_message(ChatMessage(role="assistant", content=msg))
        return
    
    if profile:
        app.current_profile = profile
        app.conversation_state = ConversationState.NORMAL

        yield ProfileExtracted(from_user_profile(profile))
        extracted_msg = app.messages.format(
            MessageKey.PROFILE_EXTRACTED,
            goal=profile.goal,
            level=profile.current_level,
            time=profile.time_commitment,
        )
        yield TextChunk(extracted_msg + "\n\n")

        logger.info("Auto-generating roadmap after profile extraction")
        yield from generate_roadmap(app)
    else:
        app.conversation_state = ConversationState.NORMAL
        missing = getattr(app._profile_extractor, "last_missing_fields", [])
        missing_msg = _missing_field_prompt(app, missing)

        msg = app.messages.format(
            MessageKey.PROFILE_STILL_INCOMPLETE,
            missing_msg=missing_msg,
        )
        yield TextChunk(msg)
        app._memory.add_message(ChatMessage(role="assistant", content=msg))
        logger.info("Profile still incomplete after re-extraction attempt")

def generate_roadmap(app: AppService) -> Generator[Event, None, None]:
    """
    Generate roadmap based on current profile

    Args:
        app: AppService instance holding current_profile and roadmap service

    Yields:
        StatusUpdate: Roadmap generation loading state
        RoadmapCreated: Roadmap generated successfully
        TextChunk: Roadmap-created confirmation message
        ErrorOccurred: Validation, LLM or unexpected error during generation
    """
    logger.info(f"generating roadmap (profile={bool(app.current_profile)})")
    yield StatusUpdate("generating_roadmap", app.messages.get(MessageKey.ROADMAP_LOADING))

    try:
        history = app._get_recent_history()
        user_text = "\n".join(m.content for m in history if m.role == "user")
        duration_week = infer_duration_weeks_from_user_text(user_text)

        roadmap = app._roadmap.generate_roadmap(
            app.current_profile,
            duration_week=duration_week,
        )
        roadmap_created_msg = app.messages.get(MessageKey.ROADMAP_CREATED)
        app.current_roadmap = roadmap
        app._memory.add_message(ChatMessage(role="assistant", content=roadmap_created_msg))
        logger.info(f"roadmap done (weeks={roadmap.duration_week}, requested_duration_week={duration_week})")
        yield RoadmapCreated(from_roadmap(roadmap))
        yield TextChunk(roadmap_created_msg)
    except ValidationError as e:
        try:
            msg = app.messages.get(MessageKey(e.code))
        except (ValueError, TypeError):
            msg = str(e) if str(e) else app.messages.get(MessageKey.ROADMAP_ERROR)
        yield ErrorOccurred("validation", msg)
        app._memory.add_message(ChatMessage(role="assistant", content=msg))
    except LLMServiceError:
        msg = app.messages.get(MessageKey.LLM_ERROR)
        yield ErrorOccurred("llm", msg)
        app._memory.add_message(ChatMessage(role="assistant", content=msg))
    except Exception as e:
        logger.exception(f"Roadmap generation failed: {e}")
        msg = app.messages.get(MessageKey.UNEXPECTED_ERROR)
        yield ErrorOccurred("unexpected", msg)
        app._memory.add_message(ChatMessage(role="assistant", content=msg))