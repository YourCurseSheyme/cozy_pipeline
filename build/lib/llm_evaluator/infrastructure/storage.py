import csv
import ast
import re
import logging
from typing import Iterator, TextIO, Optional, Any
from pathlib import Path

from src.llm_evaluator.domain.models import RawInputRow, EvaluationTask
from src.llm_evaluator.domain.exceptions import DataValidationError

logger = logging.getLogger(__name__)


class ResultWriter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file: Optional[TextIO] = None
        self._csv_writer: Optional[csv.DictWriter[Any]] = None
        self._headers = [
            "task_id", "category", "raw_output",
            "predicted_index", "ground_truth", "is_correct",
            "latency_ms", "model"
        ]

    def __enter__(self) -> "ResultWriter":
        f = open(self.file_path, "w", newline="", encoding="utf-8", buffering=1)
        self._file = f
        self._csv_writer = csv.DictWriter(f, fieldnames=self._headers)
        self._csv_writer.writeheader()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._file:
            self._file.close()

    def write_result(self, result_row: dict[str, Any]) -> None:
        if self._csv_writer:
            self._csv_writer.writerow(result_row)


class CSVReader:
    def stream_input(self, file_path: str) -> Iterator[EvaluationTask]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                reader.fieldnames = [name.strip() for name in reader.fieldnames]

            logger.info(f"CSV Columns found: {reader.fieldnames}")

            for row_num, row_dict in enumerate(reader, start=1):
                try:
                    gt_raw = (
                            row_dict.get("ground_truth") or
                            row_dict.get("ground_truth_index") or
                            row_dict.get("answer")
                    )

                    raw = RawInputRow(
                        question=row_dict.get("question", ""),
                        options=row_dict.get("options", ""),
                        category=row_dict.get("category", "unknown"),
                        ground_truth=gt_raw
                    )

                    options_str = raw.options.strip()
                    options_list = []

                    regex_matches = re.findall(r"['\"](.*?)['\"]", options_str, re.DOTALL)

                    try:
                        ast_parsed = ast.literal_eval(options_str)
                        if isinstance(ast_parsed, list):
                            if len(ast_parsed) == 1 and len(regex_matches) > 1:
                                options_list = regex_matches
                            else:
                                options_list = ast_parsed
                    except (ValueError, SyntaxError):
                        options_list = regex_matches

                    if not options_list:
                        if "," in options_str and "[" not in options_str:
                            options_list = [opt.strip() for opt in options_str.split(",")]
                        else:
                            raise DataValidationError("Could not parse options list", context={"raw": options_str[:50]})

                    gt_index = None
                    if raw.ground_truth:
                        clean_gt = str(raw.ground_truth).strip()
                        if "." in clean_gt:
                            clean_gt = clean_gt.split(".")[0]

                        if clean_gt in options_list:
                            gt_index = options_list.index(clean_gt)
                        elif clean_gt.isdigit():
                            idx = int(clean_gt)
                            if 0 <= idx < len(options_list):
                                gt_index = idx
                            else:
                                logger.warning(
                                    f"Row {row_num}: GT Index {idx} out of bounds. "
                                    f"Options count: {len(options_list)}"
                                )

                    yield EvaluationTask(
                        question=raw.question,
                        options=options_list,
                        category=raw.category,
                        ground_truth_index=gt_index
                    )

                except Exception as e:
                    logger.error(f"Skipping row {row_num}: {e}")
                    continue
