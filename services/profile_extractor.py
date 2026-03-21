"""
Extract UserProfile from chat history via LLM

Key features:
- extract(history) -> UserProfile or None
- Required fields: goal, current_level, time_commitment
- Optional fields: learning_style, background, constraints
- Tracks missing fields in last_missing_fields for specific error messages
"""
import json
import re
from typing import List

from ai import LLMClient, PROFILE_EXTRACT_PROMPT
from domain import ChatMessage, UserProfile
from utils import logger, LLMServiceError
from services.support.profile_inference import (
    infer_goal_from_user_text,
    infer_level_from_user_text,
    infer_time_commitment_from_user_text,
    resolve_goal,
    resolve_level,
    resolve_time_commitment,
)

def _history_to_text(messages: List[ChatMessage]) -> str:
    """
    Convert chat history to text for LLM prompt

    Includes both USER and ASSISTANT messages for context, but extraction focuses on USER messages

    Args:
        messages: List of ChatMessage objects

    Returns:
        Formatted text with role prefixes (User: ..., Assistant: ...)
    """
    parts = []
    for m in messages:
        if m.role == "user":
            parts.append(f"User: {m.content}")
        else:
            parts.append(f"Assistant: {m.content}")
    return "\n".join(parts)

def _parse_profile_json(raw: str) -> dict | None:
    """
    Parse JSON object from LLM response

    Handles common LLM output formats:
    - Markdown code block: ```json {...} ```
    - Plain JSON: {...}

    Args:
        raw: Raw LLM response text

    Returns:
        Parsed dict if valid JSON found, None otherwise
    """
    raw = (raw or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", raw)

    if match:
        raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None

class ProfileExtractor:
    """
    Extract UserProfile from chat history via LLM
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize ProfileExtractor with LLM client

        Args:
            llm_client: LLM client for profile extraction
        """
        self._llm = llm_client
        self.last_missing_fields: List[str] = []

    def extract(self, history: List[ChatMessage]) -> UserProfile | None:
        """
        Extract UserProfile from chat history using rule-based inference then LLM fallback

        Args:
            history: Full chat history including user and assistant messages

        Returns:
            UserProfile when required fields (goal, current_level, time_commitment) are complete; otherwise None

        Raises:
            LLMServiceError: If the underlying LLM call fails at service level
        """
        if not history:
            return None
        
        user_message = [m for m in history if m.role == "user"]
        if not user_message:
            return None
        
        user_text = "\n".join(m.content for m in user_message)

        fast_goal = infer_goal_from_user_text(user_text)
        fast_level = infer_level_from_user_text(user_text)
        fast_time = infer_time_commitment_from_user_text(user_text)

        if fast_goal and fast_level and fast_time:
            self.last_missing_fields = []
            logger.info(
                f"Profile extracted by rule: goal={fast_goal}, level={fast_level}, time={fast_time}"
            )
            return UserProfile(
                goal=fast_goal,
                current_level=fast_level,
                time_commitment=fast_time,
            )
        
        conversation = _history_to_text(history)
        prompt = PROFILE_EXTRACT_PROMPT.replace("{history}", conversation)

        try:
            response = self._llm.generate_text(prompt)
        except LLMServiceError:
            raise
        except Exception as e:
            logger.warning(f"Profile extract from history failed: {e}")
            return None
        
        data = _parse_profile_json(response)
        if not data or not isinstance(data, dict):
            logger.warning("LLM returned empty or invalid JSON")
            data = {}
        
        goal = resolve_goal(data.get("goal"), user_text)
        level_raw = (data.get("current_level") or data.get("level") or "").strip()
        level = resolve_level(level_raw, user_text)
        time_commitment = resolve_time_commitment(data.get("time_commitment"), user_text)

        missing_fields = []
        if not goal:
            missing_fields.append("goal")
        if level not in ("beginner", "intermediate", "advanced"):
            missing_fields.append("level")
        if not time_commitment:
            missing_fields.append("time")

        if missing_fields:
            self.last_missing_fields = missing_fields
            logger.info(f"Missing {missing_fields}")
            return None
        
        learning_style = (data.get("learning_style") or "").strip() or None
        background = (data.get("background") or "").strip() or None
        constraints = data.get("constraints")
        if constraints and isinstance(constraints, list):
            constraints = [c.strip() for c in constraints if isinstance(c, str) and c.strip()]
            constraints = constraints or None
        else:
            constraints = None

        optional_info = []
        if learning_style:
            optional_info.append(f"style={learning_style}")
        if background:
            optional_info.append(f"background={background}")
        if constraints:
            optional_info.append(f"constraints={constraints}")

        optional_str = f", {', '.join(optional_info)}" if optional_info else ""
        logger.info(f"goal={goal}, level={level}, time={time_commitment}{optional_str}")

        return UserProfile(
            goal=goal,
            current_level=level,
            time_commitment=time_commitment,
            learning_style=learning_style,
            background=background,
            constraints=constraints,
        )