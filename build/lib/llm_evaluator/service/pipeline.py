import asyncio
import logging
from typing import Set

from tqdm import tqdm

from src.llm_evaluator.config import settings
from src.llm_evaluator.domain.models import EvaluationTask, EvaluationResult
from src.llm_evaluator.infrastructure.storage import CSVReader, ResultWriter
from src.llm_evaluator.infrastructure.ollama import OllamaClient
from src.llm_evaluator.service.scorer import ResponseScorer

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    def __init__(self) -> None:
        self.reader = CSVReader()

    async def process_single_task(
            self,
            client: OllamaClient,
            task: EvaluationTask,
    ) -> EvaluationResult:
        try:
            llm_response = await client.generate(task.formatted_prompt)
            parsed_index = ResponseScorer.parse_index(llm_response.raw_content)
            is_correct = ResponseScorer.evaluate(
                parsed_index,
                task.ground_truth_index,
            )

            return EvaluationResult(
                task_id=task.id,
                category=task.category,
                raw_llm_output=llm_response.raw_content,
                parsed_index=parsed_index,
                ground_truth_index=task.ground_truth_index,
                is_correct=is_correct,
                latency_ms=llm_response.client_latency_ms,
                model_name=settings.MODEL_NAME
            )
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            return EvaluationResult(
                task_id=task.id,
                category=task.category,
                raw_llm_output=f"ERROR: {str(e)}",
                parsed_index=-1,
                ground_truth_index=task.ground_truth_index,
                is_correct=False,
                latency_ms=0.0,
                model_name=settings.MODEL_NAME
            )

    async def run(self, input_file: str, output_file: str) -> None:
        logger.info(f"Starting evaluation: {input_file} -> {output_file}")

        total_processed = 0
        total_correct = 0

        async with OllamaClient() as client:
            with ResultWriter(output_file) as writer:
                pending_tasks: Set[asyncio.Task[EvaluationResult]] = set()
                row_iterator = self.reader.stream_input(input_file)
                pbar = tqdm(desc="Evaluating", unit="row")

                def handle_result(res: EvaluationResult) -> None:
                    nonlocal total_processed, total_correct
                    writer.write_result(res.to_csv_row())
                    pbar.update(1)

                    if res.ground_truth_index is not None:
                        total_processed += 1
                        if res.is_correct:
                            total_correct += 1

                try:
                    for task_data in row_iterator:
                        if len(pending_tasks) >= settings.CONCURRENCY_LIMIT:
                            done, pending_tasks = await asyncio.wait(
                                pending_tasks,
                                return_when=asyncio.FIRST_COMPLETED,
                            )

                            for finished_task in done:
                                handle_result(await finished_task)

                        coro = self.process_single_task(client, task_data)
                        task = asyncio.create_task(coro)
                        pending_tasks.add(task)

                    if pending_tasks:
                        done, _ = await asyncio.wait(
                            pending_tasks,
                            return_when=asyncio.ALL_COMPLETED
                        )
                        for finished_task in done:
                            handle_result(await finished_task)
                finally:
                    if pending_tasks:
                        for t in pending_tasks:
                            t.cancel()
                    pbar.close()

            if total_processed > 0:
                accuracy = (total_correct / total_processed) * 100
                logger.info("=" * 40)
                logger.info(f"Evaluation Complete")
                logger.info(f"Total Processed: {total_processed}")
                logger.info(f"Total Correct:   {total_correct}")
                logger.info(f"Accuracy:        {accuracy:.2f}%")
                logger.info("=" * 40)
            else:
                logger.warning("Evaluation complete but no rows were processed.")
