from http import HTTPStatus
from typing import Any, Optional, Dict

class LearnPathException(Exception):
    code: str = "GENERAL_ERROR"
    message: str = "An unexpected error occurred"
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value

    def __init__(
        self,
        message: Optional[str] = None,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs: Any,
    ):
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.extra = kwargs
        
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "status_code": self.status_code,
                **self.extra,
            }
        }
    
    def __str__(self) -> str:
        return f"{self.code} ({self.status_code}): {self.message}"

class LLMServiceError(LearnPathException):
    code = "LLM_SERVICE_ERROR"
    message = "Failed to communicate with the LLM service"
    status_code = HTTPStatus.BAD_REQUEST.value

class ValidationError(LearnPathException):
    code = "VALIDATION_ERROR"
    message = "Invalid input data provided"
    status_code = HTTPStatus.BAD_REQUEST.value