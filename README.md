# LearnPath Chatbot

LearnPath Chatbot is an AI-powered learning-path assistant that uses Google Gemini to understand user conversations, extract user profile information, and generate personalized study roadmaps.

## Overview

- Streamlit-based UI (`app.py`)
- Service layer for chat flow, intent detection, profile extraction, and roadmap generation
- Real Gemini integration via `google-generativeai` with streaming support
- Centralized configuration via `pydantic-settings` and `.env`
- Unit and integration test suites powered by `pytest`

## Project Structure

```text
learn_path_ai/
|- ai/            # Gemini client, prompts, and LLM abstractions
|- config/        # Settings, constants, message providers
|- domain/        # Domain models and events
|- memory/        # Conversation history management (in-memory)
|- services/      # Core business logic: app/chat/session/intent/profile/roadmap
|- ui/            # Streamlit UI components
|- utils/         # Retry, logger, exceptions
|- tests/         # Unit tests + integration tests + fixtures
|- app.py         # Streamlit entrypoint
|- requirements.txt
```

## Requirements

- Python >= 3.10
- A valid Gemini API key

> Note: this project uses `X | None` type hints, which require Python 3.10+.

## Installation

```bash
git clone https://github.com/Hoaithuong3953/learn_path_ai.git
cd learn_path_ai
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_MODEL=gemini-2.5-flash

LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s
LOG_DATE_FORMAT=%Y-%m-%d %H:%M:%S
LOG_TO_FILE=false
LOG_FILE_PATH=logs/app.log
LOG_FILE_ROTATION=midnight
LOG_FILE_RETENTION=7
```

## Run the App

```bash
streamlit run app.py
```

After launch, open the Streamlit URL shown in terminal (typically `http://localhost:8501`).

## Run Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov
```

Test structure:
- `tests/unit/`: module-level unit tests
- `tests/integration/`: end-to-end chat/roadmap flow integration tests
- `tests/fixtures/`: sample test data

## Main Technologies

- `streamlit`
- `google-generativeai`
- `pydantic`, `pydantic-settings`
- `tenacity`
- `pytest`, `pytest-cov`