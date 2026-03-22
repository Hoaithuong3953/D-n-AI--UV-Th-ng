"""
test_prompts.py

Smoke tests for prompt constants and Template substitution used by services
"""
from ai.prompts import PROFILE_EXTRACT_PROMPT, ROADMAP_PROMPT_TEMPLATE, SYSTEM_PROMPT

def test_system_prompt_is_non_empty_and_branded():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT.strip()) > 0
    assert "LearnPath" in SYSTEM_PROMPT

def test_roadmap_prompt_template_substitute_matches_roadmap_service_usage():
    """Same keys as roadmap_service._build_prompt (duration_week as string)"""
    text = ROADMAP_PROMPT_TEMPLATE.substitute(
        goal="Học X",
        level="beginner",
        time_commitment="1 giờ",
        learning_style="video",
        background="none",
        constraints="none",
        duration_week="8",
    )
    assert "Học X" in text
    assert "beginner" in text
    assert "8" in text
    assert "Dựa trên thông tin sau" in text

def test_profile_extract_prompt_history_placeholder():
    """profile_extractor replaces {history} with conversation text"""
    assert "{history}" in PROFILE_EXTRACT_PROMPT
    filled = PROFILE_EXTRACT_PROMPT.replace("{history}", "User: hello")
    assert "User: hello" in filled
    assert "{history}" not in filled