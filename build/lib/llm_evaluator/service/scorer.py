import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ResponseScorer:
    _ANSWER_PATTERN = re.compile(r"ANSWER\s*[:\-]?\s*(\d+)", re.IGNORECASE)

    @staticmethod
    def parse_index(raw_output: str) -> int:
        if not raw_output:
            return -1

        match = ResponseScorer._ANSWER_PATTERN.search(raw_output)
        if match:
            return int(match.group(1))

        digits = re.findall(r"\b(\d+)\b", raw_output)
        if digits:
            return int(digits[-1])

        return -1

    @staticmethod
    def evaluate(parsed_index: int, ground_truth_index: Optional[int]) -> bool:
        if ground_truth_index is None:
            return False
        return parsed_index == ground_truth_index
