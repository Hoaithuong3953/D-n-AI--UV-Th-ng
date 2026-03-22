"""
Integration tests: AppService + fake LLM (``tests/integration/conftest.py``).

Layout mirrors ``services/flows/``:
- ``chat_flow/`` — streaming chat, session, memory, errors
- ``roadmap_flow/`` — ROADMAP intent, profile, roadmap generation, profile
  extract / re-extract failures, ``reset_session`` domain state
"""
