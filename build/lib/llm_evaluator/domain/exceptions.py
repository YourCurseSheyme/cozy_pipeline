from typing import Any, Optional

class LLMEvaluatorError(Exception):
    def __init__(self, message: str, context: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.context:
            return f"{base_msg} | Context: {self.context}"
        return base_msg

class DataValidationError(LLMEvaluatorError):
    pass

class LLMConnectionError(LLMEvaluatorError):
    pass

class ParsingError(LLMEvaluatorError):
    pass
