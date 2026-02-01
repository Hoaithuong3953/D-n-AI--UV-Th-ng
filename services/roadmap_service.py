"""
Roadmap service for LearnPath chatbot

Builds prompts from UserProfile, calls the LLM to generate raw JSON roadmaps,
parses them into domain Roadmap objects and applies retry policy plus validation rules
"""

import json
from typing import Optional

from ai import LLMClient, ROADMAP_PROMPT_TEMPLATE
from domain import Roadmap, UserProfile
from utils import LLMServiceError, ValidationError, logger

class RoadmapService:
    """
    Service for generating and validating learning roadmaps

    Responsibilities:
    - Build roadmap generation prompt from UserProfile
    - Call LLMClient.generate_text to obtain raw JSON
    - Parse JSON into Roadmap domain model
    - Apply retry-on-invalid-output policy
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_retries: int = 2,
    ):
        self.llm = llm_client
        self.max_retries = max_retries

    def generate_roadmap(
        self,
        profile: UserProfile,
        duration_week: Optional[int] = None
    ) -> Roadmap:
        """
        Generate a Roadmap from a UserProfile

        Args:
            profile: Collected user profile information
            duration_week: Optional override for total duration in weeks
        
        Returns:
            Roadmap domain object

        Raises:
            ValidationError: If after max_retries the LLM output is still invalid
            LLMServiceError: Propagated if underlying LLM call fails permanently
        """
        duration = duration_week or self._guess_duration(profile)

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            prompt = self._build_prompt(
                profile = profile,
                duration_week = duration
            )

            try:
                raw = self.llm.generate_text(prompt)
                roadmap = self._parse_and_validate(raw)
                logger.info(f"Roadmap generation succeeded on attempt {attempt}")
                return roadmap
            except (ValidationError, LLMServiceError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Roadmap generation attempt {attempt} failed: {e}"
                )
                last_error = e

        message = (
            "Không thể tạo lộ trình học tập hợp lệ sau khi thử lại nhiều lần."
            "Vui lòng thử lại hoặc điều chỉnh thông tin đầu vào."
        )
        raise ValidationError(message=message, code="ROADMAP_GENERATION_FAILED") from last_error
        
    def _build_prompt(self, profile: UserProfile, duration_week: int) -> str:
        """Build roadmap generation prompt using ROADMAP_PROMPT_TEMPLATE"""
        learning_style = profile.learning_style or "Không cung cấp"
        background = profile.background or "Không cung cấp"
        constraints = ", ".join(profile.constraints or ["Không có"])

        prompt = ROADMAP_PROMPT_TEMPLATE.substitute(
            goal=profile.goal,
            level=profile.current_level,
            time_commitment=profile.time_commitment,
            learning_style=learning_style,
            background=background,
            constraints=constraints,
            duration_week=str(duration_week),
        )

        return prompt
    
    def _parse_and_validate(self, raw_json: str) -> Roadmap:
        """Parse LLM JSON output and validate against Roadmap schema"""
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode roadmap JSON: {e}")
            raise ValidationError(message="LLM trả về JSON không hợp lệ") from e
        
        try:
            roadmap = Roadmap.model_validate(data)
        except Exception as e:
            logger.error(f"Roadmap validation failed: {e}")
            raise ValidationError(message="Roadmap không hợp lệ theo schema") from e
        
        return roadmap
    
    def _guess_duration(self, profile: UserProfile) -> int:
        """Simple heuristic to guess duration_week from profile"""
        return 8