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

from ai import LLMClient
from domain import ChatMessage, UserProfile
from utils import logger, LLMServiceError

PROFILE_EXTRACT_PROMPT = """
Từ đoạn hội thoại sau, trích xuất thông tin hồ sơ học tập từ tin nhắn của USER
Trả về ĐÚNG MỘT object JSON với các key sau:

REQUIRED (bắt buộc - phải có rõ ràng trong tin nhắn USER):
- goal: string (mục tiêu học tập, vd: "Học Python", "Lập trình web")
current_level: một trong "beginner", "intermediate", "advanced"
- time_commitment: string (thời gian mỗi ngày, vd: "30 phút", "1 giờ", "2 giờ")

OPTIONAL (nếu có thông tin từ USER):
- learning_style: string (phong cách học tập, vd: "Học qua video", "Đọc tài liệu", "Thực hành")
- background: string (nền tảng/kinh nghiệm trước đó, vd: "Đã học HTML/CSS", "Chưa biết lập trình")
- constraints: array of strings (các ràng buộc, vd: ["Chỉ tài liệu miễn phí", "Học vào cuối tuần"])

QUY TẮC QUAN TRỌNG:
1. CHỈ trích xuất thông tin từ tin nhắn của USER, BỎ QUA tin nhắn của Assistant
2. KHÔNG bịa hoặc suy đoán thông tin không có trong tin nhắn USER
3. Nếu USER KHÔNG nói rõ required field nào → trả về {}
4. Nếu có thông tin optional thì thêm vào JSON, không có thì bỏ qua (đừng thêm null/None)
5. Extract CHÍNH XÁC từ nguyên văn, không paraphrase nếu không cần thiết

FORMAT MẪU (chỉ để tham khảo):

Ví dụ 1 - Đầy đủ thông tin:
Hội thoại:
User: "Tôi muốn học Python cơ bản, mình mới bắt đầu, có khoảng 2 giờ/ngày"
User: "Tôi thích học qua video và đã biết HTML/CSS rồi"
Assistant: "Tốt, tôi sẽ tạo lộ trình..."

JSON:
{
    "goal": "Học Python cơ bản",
    "current_level": "beginner",
    "time_commitment": "2 giờ",
    "learning_style": "Học qua video",
    "background": "Đã biết HTML/CSS"
}

Ví dụ 2 - Chỉ có required:
Hội thoại:
User: "Học JavaScript cho người mới, 1 giờ mỗi ngày"
Assistant: "Bạn có muốn học qua video không?"
User: "Tùy cũng được"

JSON:
{
    "goal": "Học JavaScript",
    "current_level": "beginner",
    "time_commitment": "1 giờ"
}

Ví dụ 3 - Thiếu time_commitment (KHÔNG đủ required):
Hội thoại:
User: "Tôi muốn học React, mình intermediate rồi"
Assistant: "Bạn có bao nhiêu thời gian mỗi ngày?"
User: "Chưa biết, linh hoạt"

JSON:
{}

Ví dụ 4 - Assistant nói sai, USER sửa (CHỈ lấy từ USER):
Hội thoại:
User: "Tôi muốn học Python"
Assistant: "Bạn muốn học advanced phải không?"
User: "Không, mình mới bắt đầu mà, beginner. Có 1 giờ/ngày"

JSON:
{
    "goal": "Học Python",
    "current_level": "beginner",
    "time_commitment": "1 giờ"
}
(KHÔNG extract "advanced" từ Assistant message!)

Ví dụ 5 - USER nói mơ hồ về level (KHÔNG bịa):
Hội thoại:
User: "Tôi muốn học Python, có 2 giờ/ngày"
Assistant: "Bạn đang ở level nào?"
User: "Cũng biết chút chút"

JSON:
{}
(USER không nói rõ "beginner"/"intermediate"/"advanced" -> KHÔNG đủ required!)

Hội thoại (chỉ xét các tin nhắn gần đây):
---
{history}
---
JSON:
"""

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
        return json.load(raw)
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
        
        """
        if not history:
            return None
        
        user_message = [m for m in history if m.role == "user"]
        if not user_message:
            return None
        
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
            self.last_missing_fields = ["goal", "level", "time"]
            return None
        
        goal = (data.get("goal") or "").strip()
        level = (data.get("level") or "").strip()
        time_commitment = (data.get("time_commitment") or "").strip()

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