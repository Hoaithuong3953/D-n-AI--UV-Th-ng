"""
test_resource.py

Unit tests for Resource model (title, url, type, difficulty, validation)
"""
import pytest
from pydantic import ValidationError

from domain import Resource

class TestResource:
    """Tests for Resource model"""

    def test_create_resource_minimal(self):
        """Resource with minimal required fields is valid"""
        resource = Resource(
            title="Python Tutorial",
            url="https://example.com/python",
            type="documentation"
        )

        assert resource.title == "Python Tutorial"
        assert str(resource.url) == "https://example.com/python"
        assert resource.type == "documentation"
        assert resource.description is None
        assert resource.difficulty is None

    def test_create_resource_full(self):
        """Resource with all fields (description, difficulty) is valid"""
        resource = Resource(
            title="Python Crash Course",
            url="https://example.com/crash-course",
            type="video",
            description="Comprehensive Python tutorial",
            difficulty="beginner"
        )

        assert resource.description == "Comprehensive Python tutorial"
        assert resource.difficulty == "beginner"

    def test_resource_url_validation(self):
        """url must be valid HttpUrl; invalid raises ValidationError"""
        # Valid URL
        resource = Resource(
            title="Test",
            url="https://example.com",
            type="documentation"
        )

        assert str(resource.url) == "https://example.com/"

        # Invalid URL should raise ValidationError
        with pytest.raises(ValidationError):
            Resource(
                title="Test",
                url="not a url",
                type="documentation"
            )

    def test_resource_type_literal(self):
        """type must be one of video, article, book, course, practice, project, documentation"""
        # Valid types
        valid_types = ["video", "article", "book", "course", "practice", "project", "documentation"]
        for resource_type in valid_types:
            resource = Resource(
                title="Test",
                url="https://example.com",
                type=resource_type
            )
            assert resource.type == resource_type

        # Invalid type should raise ValidationError
        with pytest.raises(Exception):
            Resource(
                title="Test",
                url="https://test.com",
                type="invalid_type"
            )

    def test_resource_difficulty_literal(self):
        """difficulty if provided must be beginner, intermediate, or advanced; invalid raises ValidationError"""
        # Valid difficulties
        valid_difficulties = ["beginner", "intermediate", "advanced"]
        for difficulty in valid_difficulties:
            resource = Resource(
                title="Test",
                url="https://example.com",
                type="documentation",
                difficulty=difficulty
            )

            assert resource.difficulty == difficulty

        # Invalid difficulty should raise ValidationError
        with pytest.raises(ValidationError):
            Resource(
                title="Test",
                url="https://test.com",
                type="documentation",
                difficulty="invalid"
            )

    def test_resource_title_max_length(self):
        """title max_length (200) is enforced; excess raises ValidationError"""
        # Valid length
        resource = Resource(
            title="A" * 200,
            url="https://example.com",
            type="documentation"
        )
        assert len(resource.title) == 200

        # Exceeds max_length should raise ValidationError
        with pytest.raises(ValidationError):
            Resource(
                title="A" * 201,
                url="https://example.com",
                type="documentation"
            )
