import uuid
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

from src.llm_evaluator.config import settings


class RawInputRow(BaseModel):
    question: str
    options: str
    category: str
    ground_truth: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class EvaluationTask(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    question: str
    options: List[str]
    category: str
    ground_truth_index: Optional[int] = None

    model_config = ConfigDict(frozen=True)

    @property
    def formatted_prompt(self) -> str:
        options_text = "\n".join(
            [f"[{i}] {opt}" for i, opt in enumerate(self.options)]
        )

        cat_key = self.category.lower().strip() if self.category else "default"
        driver = settings.KERNEL_DRIVERS.get(
            cat_key,
            settings.KERNEL_DRIVERS["default"]
        )

        prompt = (
            f"{settings.KERNEL_BOOTLOADER}\n\n"
            f"{driver}\n\n"
            f"Question: {self.question}\n"
            f"Options:\n{options_text}\n\n"
        )

        return prompt

class LLMResponse(BaseModel):
    raw_content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    client_latency_ms: float = 0.0

class EvaluationResult(BaseModel):
    task_id: uuid.UUID
    category: str
    raw_llm_output: str
    parsed_index: int
    ground_truth_index: Optional[int]
    is_correct: bool
    latency_ms: float
    model_name: str
    timestamp: float = Field(default_factory=lambda: 0.0)

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "category": self.category,
            "raw_output": self.raw_llm_output,
            "predicted_index": self.parsed_index,
            "ground_truth": self.ground_truth_index,
            "is_correct": self.is_correct,
            "latency_ms": round(self.latency_ms, 2),
            "model": self.model_name
        }
