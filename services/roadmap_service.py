"""
roadmap_service.py

Roadmap service for LearnPath chatbot. Builds prompts from UserProfile, calls LLM
for raw JSON roadmaps, parses into Roadmap domain model and applies retry/validation

Key features:
- generate_roadmap with retry logic (max 2 attempts on validation failure)
- Supports roadmap adjustment (provide previous_roadmap + chat_context)
- Validates output against Roadmap schema (duration, milestones, resources)
"""
import json
from typing import Optional

from ai import LLMClient, ROADMAP_PROMPT_TEMPLATE
from domain import Roadmap, UserProfile
from utils import LLMServiceError, ValidationError, logger
from config import MessageKey

class RoadmapService:
    """
    Generate and validate learning roadmaps from user profiles

    Responsibilities:
    - Build roadmap generation prompt from UserProfile
    - Call LLMClient.generate_text to obtain raw JSON
    - Parse JSON into Roadmap domain model; apply retry on invalid output
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_retries: int = 2,
    ):
        """
        Initialize with LLM client and max retries for roadmap generation

        Args:
            llm_client: LLM client for roadmap generation
            max_retries: Maximum retry attempts on validation failure (default: 2)
        """
        self.llm = llm_client
        self.max_retries = max_retries

    def generate_roadmap(
        self,
        profile: UserProfile,
        duration_week: Optional[int] = None,
        chat_context: Optional[str] = None,
        previous_roadmap: Optional[Roadmap] = None,
    ) -> Roadmap:
        """
        Generate  or adjust a Roadmap from a UserProfile

        Args:
            profile: Collected user profile information
            duration_week: Optional override for total duration in weeks
            chat_context: Optional recent chat for personalization
            previous_roadmap: Optional existing roadmap for adjust
        
        Returns:
            Roadmap domain object

        Raises:
            ValidationError: If after max_retries the LLM output is still invalid
            LLMServiceError: Propagated if underlying LLM call fails permanently
        """
        duration = duration_week or self._guess_duration(profile)
        last_error: Optional[Exception] = None
        logger.info(f"generate_roadmap start for profile: {profile.model_dump()} with duration_week: {duration_week}")

        for attempt in range(1, self.max_retries + 1):
            if attempt > 1:
                logger.info(f"generate_roadmap retry attempt {attempt}")
            prompt = self._build_prompt(
                profile = profile,
                duration_week = duration,
                chat_context = chat_context,
                previous_roadmap = previous_roadmap,
            )

            try:
                raw = self.llm.generate_text(prompt)
                roadmap = self._parse_and_validate(raw)
                logger.info(f"Roadmap generation succeeded on attempt {attempt}")
                return roadmap
            except (ValidationError, LLMServiceError, json.JSONDecodeError) as e:
                logger.warning(f"Roadmap generation attempt {attempt} failed: {e}")
                last_error = e

        raise ValidationError(code=MessageKey.ROADMAP_GENERATION_FAILED.value) from last_error
        
    def _build_prompt(
        self,
        profile: UserProfile,
        duration_week: int,
        chat_context: str = "",
        previous_roadmap: Optional[Roadmap] = None,
    ) -> str:
        """
        Build roadmap generation prompt from profile and context
        
        Args:
            profile: User profile
            duration_week: Total weeks for roadmap
            chat_context: Recent chat context for personalization (optional)
            previous_roadmap: Existing roadmap to adjust (optional)

        Returns:
            Formatted prompt string ready for LLM
        """
        learning_style = profile.learning_style or "Không cung cấp"
        background = profile.background or "Không cung cấp"
        constraints = ", ".join(profile.constraints or ["Không có"])

        if chat_context and chat_context.strip():
            chat_context_section = (
                "Ngữ cảnh hội thoại gần đây (ưu tiên theo yêu cầu trong đó): \n"
                + chat_context.strip() + "\n\n"
            )
        else:
            chat_context_section = ""

        if previous_roadmap:
            roadmap_json = json.dumps(
                previous_roadmap.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            previous_roadmap_section = (
                "LỘ TRÌNH HIỆN TẠI (điều chỉnh theo yêu cầu mới trong hội thoại trên):\n"
                + roadmap_json 
                + "\n\n"
                "Hãy ĐIỀU CHỈNH lộ trình trên theo yêu cầu mới (vd: làm ngắn hơn, thêm tuần, đổi tài liệu)."
                "Output vẫn là JSON đúng cấu trúc, duration_week và số milestone phải khớp.\n\n"
            )
        else:
            previous_roadmap_section = ""

        prompt = ROADMAP_PROMPT_TEMPLATE.substitute(
            goal=profile.goal,
            level=profile.current_level,
            time_commitment=profile.time_commitment,
            learning_style=learning_style,
            background=background,
            constraints=constraints,
            duration_week=str(duration_week),
            chat_context_section=chat_context_section,
            previous_roadmap_section=previous_roadmap_section,
        )

        return prompt
    
    def _parse_and_validate(self, raw_json: str) -> Roadmap:
        """
        Parse LLM JSON output and validate against Roadmap schema
        
        Args:
            raw_json: Raw JSON string from LLM

        Returns:
            Validated Roadmap domain object

        Raises:
            ValidationError: JSON parse error (ROADMAP_INVALID_JSON) or schema validation error (ROADMAP_INVALID_SCHEMA)
        """
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode roadmap JSON: {e}")
            raise ValidationError(code=MessageKey.ROADMAP_INVALID_JSON.value) from e
        
        try:
            roadmap = Roadmap.model_validate(data)
        except Exception as e:
            logger.error(f"Roadmap validation failed: {e}")
            raise ValidationError(code=MessageKey.ROADMAP_INVALID_SCHEMA.value) from e
        
        return roadmap
    
    def _guess_duration(self) -> int:
        """Guess duration_week from profile (simple heuristic)"""
        return 8