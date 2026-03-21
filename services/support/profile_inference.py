from __future__ import annotations

import re

GOAL_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Python", ("python", "django", "flask", "fastapi")),
    ("JavaScript", ("javascript", "js", "nodejs", "node.js")),
    ("TypeScript", ("typescript",)),
    ("React", ("react",)),
    ("Java", ("java", "spring", "spring boot")),
    ("C++", ("c++", "cpp")),
    ("C#", ("c#", "scharp", ".net", "dotnet")),
    ("SQL", ("sql", "postgres", "mysql", "sqlite")),
)

BEGINNER_LEVEL_MARKERS: tuple[str, ...] = (
    "beginner", "tu dau", "từ đầu", "moi bat dau", "mới bắt đầu",
    "co ban", "cơ bản", "nguoi moi", "người mới", "chua biet", "chưa biết",
)

INTERMEDIATE_LEVEL_MARKERS: tuple[str, ...] = (
    "intermediate", "da co kinh nghiem", "đã có kinh nghiệm",
    "co kinh nghiem", "có kinh nghiệm", "co nen", "có nền",
    "co nen tang", "có nền tảng", "kha", "khá",
)

ADVANCED_LEVEL_MARKERS: tuple[str, ...] = (
    "advanced", "nang cao", "nâng cao", "chuyen sau", "chuyên sâu",
    "thanh thao", "thành thạo", "professional", "expert",
)

def infer_goal_from_user_text(user_text: str) -> str | None:
    """Infer a concrete goal from user text (minimal keyword mapping)"""
    t = (user_text or "").lower()
    for goal, markers in GOAL_MARKERS:
        if any(marker in t for marker in markers):
            return goal
    return None

def infer_level_from_user_text(user_text: str) -> str | None:
    """Infer normalized level from user text"""
    t = (user_text or "").lower()

    if any(m in t for m in BEGINNER_LEVEL_MARKERS):
        return "beginner"
    if any(m in t for m in INTERMEDIATE_LEVEL_MARKERS):
        return "intermediate"
    if any(m in t for m in ADVANCED_LEVEL_MARKERS):
        return "advanced"
    return None

def infer_time_commitment_from_user_text(user_text: str) -> str | None:
    """Infer daily study time from common Vietnamese/English patterns"""
    text = (user_text or "").lower()

    hour_patterns = [
        r"(\d+(?:[.,]\d+)?)\s*(?:giờ|gio|h)\s*(?:\/\s*ngày|mỗi\s*ngày|mot\s*ngay|per\s*day|/day)",
        r"(?:mỗi\s*ngày|per\s*day)\s*(\d+(?:[.,]\d+)?)\s*(?:giờ|gio|h)",
    ]
    for pattern in hour_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            value = m.group(1).replace(",", ".")
            if value.endswith(".0"):
                value = value[:-2]
            return f"{value} giờ/ngày"
        
    minute_patterns = [
        r"(\d{1,3})\s*(?:phút|phut)\s*(?:\/\s*ngày|mỗi\s*ngày|per\s*day|/day)",
        r"(?:mỗi\s*ngày|per\s*day)\s*(\d{1,3})\s*(?:phút|phut)",
    ]
    for pattern in minute_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1)} phút/ngày"
    return None

def infer_duration_weeks_from_user_text(user_text: str) -> int | None:
    """Infer total roadmap duration in weeks from user text"""
    t = (user_text or "").lower()

    month_patterns = [
        r"(\d{1,2})\s*(?:tháng|thang|month|months)\b"
    ]
    for pattern in month_patterns:
        m = re.search(pattern, t, flags=re.IGNORECASE)
        if m:
            months = int(m.group(1))
            if months > 0:
                return months * 4
            
    week_patterns = [
        r"(\d{1,2})\s*(?:tuần|tuan|week|weeks)\b"
    ]
    for pattern in week_patterns:
        m = re.search(pattern, t, flags=re.IGNORECASE)
        if m:
            weeks = int(m.group(1))
            if weeks > 0:
                return weeks
    return None

def resolve_goal(raw_goal: str | None, user_text: str) -> str:
    """Merge LLM ``goal`` with rule-based inference from ``user_text``"""
    g = (raw_goal or "").strip()
    if g:
        return g
    return infer_goal_from_user_text(user_text) or ""

def resolve_level(raw_level: str | None, user_text: str) -> str | None:
    """Merge LLM ``current_level`` from rule-based inference"""
    from_user = infer_level_from_user_text(user_text)
    if from_user:
        return from_user
    
    if not raw_level:
        return None
    
    level = raw_level.strip().lower()
    if level in ("beginner", "intermediate", "advanced"):
        return level
    return infer_level_from_user_text(level)

def resolve_time_commitment(raw: str | None, user_text: str) -> str:
    """Merge LLM ``time_commitment`` with regex inference from ``user_text``"""
    t = (raw or "").strip()
    if t:
        return t
    return infer_time_commitment_from_user_text(user_text) or ""