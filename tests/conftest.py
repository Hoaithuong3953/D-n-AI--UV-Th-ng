"""
Pytest fixtures for LearnPath chatbot tests
"""
import pytest
from unittest.mock import MagicMock, patch

from domain import ChatMessage, UserProfile, Roadmap, Milestone, Resource

@pytest.fixture
def mock_genai_model():
    """Mock google.generativeai for GeminiClient tests"""
    with patch("ai.gemini_client.genai.GenerativeModel") as mock_model_cls, patch("ai.gemini_client.genai.configure",) as mock_configure:
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance
        yield mock_model_cls, mock_instance, mock_configure

@pytest.fixture
def sample_chat_message() -> ChatMessage:
    """Sample chat message for testing"""
    return ChatMessage(
        role="user",
        content="Tôi muốn học Python",
    )

@pytest.fixture
def sample_user_profile() -> UserProfile:
    """Sample user profile for testing"""
    return UserProfile(
        goal="Học python để làm data science",
        current_level="Beginner",
        time_commitment="10 giờ/tuần",
        learning_style="Visual learner",
        background="Có kinh nghiệm với Excel",
        constraints=["Thời gian hạn chế", "Ngân sách có hạn"]
    )

@pytest.fixture
def sample_user_profile_minimal() -> UserProfile:
    """Sample user profile without optional fields"""
    return UserProfile(
        goal="Học python để làm data science",
        current_level="Beginner",
        time_commitment="10 giờ/tuần",
    )

@pytest.fixture
def sample_resource() -> Resource:
    """Sample resource for testing"""
    return Resource(
        title="Python Tutorial",
        url="https://docs.python.org/3/tutorial/",
        type="documentation",
        description="Official Python tutorial",
        difficulty="beginner"
    )

@pytest.fixture
def sample_resource_minimal() -> Resource:
    """Sample resource without optional fields"""
    return Resource(
        title="Python Tutorial",
        url="https://docs.python.org/3/tutorial/",
        type="documentation"
    )

@pytest.fixture
def sample_milestone() -> Milestone:
    """Sample milestone for testing"""
    return Milestone(
        week=1,
        topic="Python Basics",
        description="Học biến, vòng lặp và cấu trúc điều khiển",
        resources=[
            Resource(
                title="Python Tutorial",
                url="https://docs.python.org/3/tutorial/",
                type="documentation"
            )
        ],
        estimated_time="5 giờ",
        learning_objectives=[
            "Hiểu về biến và kiểu dữ liệu",
            "Sử dụng vòng lặp for và while",
            "Làm việc với cấu trúc điều khiển"
        ]
    )

@pytest.fixture
def sample_milestone_minimal() -> Milestone:
    """Sample milestone without optional fields"""
    return Milestone(
        week=1,
        topic="Python Basics",
        description="Học biến, vòng lặp và cấu trúc điều khiển",
        resources=[
            Resource(
                title="Python Tutorial",
                url="https://docs.python.org/3/tutorial/",
                type="documentation"
            )
        ]
    )

@pytest.fixture
def sample_roadmap() -> Roadmap:
    """Sample roadmap for testing"""
    return Roadmap(
        topic="Học Python cơ bản",
        duration_week=4,
        milestones=[
            Milestone(week=1, topic="Python Basics", description="Học biến, vòng lặp", resources=[
                Resource(
                    title= "Python Tutorial",
                    url="https://docs.python.org/3/tutorial",
                    type="documentation"
                )
            ]),
            Milestone(week=2, topic="Functions and Modules", description="Học về functions và modules", resources=[
                Resource(
                    title= "Python Functions",
                    url="https://docs.python.org/3/tutorial/controlflow.html",
                    type="documentation"
                )
            ]),
            Milestone(week=3, topic="Data Structure", description="Lists, dictionaries, sets", resources=[
                Resource(
                    title= "Python Data Structures",
                    url="https://docs.python.org/3/tutorial/datastructures.html",
                    type="documentation"
                )
            ]),
            Milestone(week=4, topic="Object-Oriented Programming", description="Classes và objects", resources=[
                Resource(
                    title= "Python Classes",
                    url="https://docs.python.org/3/tutorial/classes.html",
                    type="documentation"
                )
            ])
        ],
        title="Lộ trình học Python từ cơ bản đến nâng cao",
        description="Lộ trình học Python trong 4 tuần cho người mới bắt đầu",
        prerequisites=["Kiến thức cơ bản về máy tính", "Có thể sử dụng terminal"]
    )

@pytest.fixture
def sample_roadmap_minimal() -> Roadmap:
    """Sample roadmap without optional fields"""
    return Roadmap(
        topic="Học Python cơ bản",
        duration_week=2,
        milestones=[
            Milestone(week=1, topic="Python Basics", description="Học biến, vòng lặp", resources=[
                Resource(
                    title= "Python Tutorial",
                    url="https://docs.python.org/3/tutorial",
                    type="documentation"
                )
            ]),
            Milestone(week=2, topic="Functions and Modules", description="Học về functions và modules", resources=[
                Resource(
                    title= "Python Functions",
                    url="https://docs.python.org/3/tutorial/controlflow.html",
                    type="documentation"
                )
            ])
        ]
    )
