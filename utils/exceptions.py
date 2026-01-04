class LearnPathException(Exception):
    pass

class LLMServiceError(LearnPathException):
    pass

class ValidationError(LearnPathException):
    pass