import logging
import asyncio
import random
import psutil
import optuna
from typing import List, Dict, Any
from statistics import mean

from src.llm_evaluator.domain.models import EvaluationTask
from src.llm_evaluator.infrastructure.ollama import OllamaClient
from src.llm_evaluator.service.scorer import ResponseScorer
from src.llm_evaluator.config import settings
from src.llm_evaluator.config import _optimized_path

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

class AutoTuner:
    def __init__(self, all_tasks: List[EvaluationTask], sample_size: int = 5, n_trials: int = 15):
        self.all_tasks = all_tasks
        self.sample_tasks = (
            random.sample(all_tasks, sample_size)
            if len(all_tasks) > sample_size
            else all_tasks
        )
        self.n_trials = n_trials
        self.concurrency_limit = settings.CONCURRENCY_LIMIT

    def _get_hardware_concurrency(self) -> int:
        try:
            physical_cores = psutil.cpu_count(logical=False)
            return physical_cores if physical_cores else 4
        except Exception:
            return 4

    async def _process_single_task(
            self,
            client: OllamaClient,
            task: EvaluationTask,
            options: Dict[str, Any],
            semaphore: asyncio.Semaphore
    ) -> dict:
        async with semaphore:
            try:
                response = await client.generate(task.formatted_prompt, options_override=options)

                parsed_index = ResponseScorer.parse_index(response.raw_content)
                is_correct = ResponseScorer.evaluate(parsed_index, task.ground_truth_index)

                return {
                    "is_correct": is_correct,
                    "latency": response.client_latency_ms
                }
            except Exception as e:
                logger.debug(f"Tuning task failed: {e}")
                return {"is_correct": False, "latency": 10000.0}

    async def _evaluate_trial_async(self, options: Dict[str, Any]) -> float:
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        tasks = []

        async with OllamaClient() as client:
            for task in self.sample_tasks:
                coro = self._process_single_task(client, task, options, semaphore)
                tasks.append(coro)

            results = await asyncio.gather(*tasks)

        correct_count = sum(1 for r in results if r["is_correct"])
        latencies = [r["latency"] for r in results]

        accuracy = correct_count / len(self.sample_tasks)
        avg_latency = mean(latencies) if latencies else 0

        score = (accuracy * 1000) - (avg_latency / 1000)
        return score

    async def optimize(self) -> Dict[str, Any]:
        logger.info(f"Starting Auto-Tuning on {len(self.sample_tasks)} sample tasks over {self.n_trials} trials...")

        num_thread = self._get_hardware_concurrency()
        loop = asyncio.get_running_loop()

        def objective_wrapper(trial: optuna.Trial) -> float:
            params = {
                "num_thread": num_thread,
                "temperature": trial.suggest_float("temperature", 0.0, 0.4, step=0.05),
                "top_p": trial.suggest_float("top_p", 0.8, 1.0, step=0.05),
                "repeat_penalty": trial.suggest_float("repeat_penalty", 1.05, 1.20, step=0.05),
                "num_ctx": settings.OLLAMA_NUM_CTX
            }
            future = asyncio.run_coroutine_threadsafe(self._evaluate_trial_async(params), loop)
            return future.result()

        def run_study():
            yet_another_study = optuna.create_study(direction="maximize")
            yet_another_study.optimize(objective_wrapper, n_trials=self.n_trials)
            return yet_another_study

        study = await loop.run_in_executor(None, run_study)

        best_params = study.best_params
        best_params["num_thread"] = num_thread

        logger.info("=" * 40)
        logger.info(f"Tuning Complete. Best Score: {study.best_value:.2f}")
        logger.info(f"Best Parameters: {best_params}")
        logger.info("=" * 40)

        return best_params

    def save_recommendations(self, params: Dict[str, Any], path: str = _optimized_path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# --- COZY-LLM AUTO-TUNED CONFIG ---\n")
                f.write(f"# Generated on: {asyncio.get_event_loop().time()}\n")
                f.write(f"# These values override defaults in .env\n\n")

                f.write(f"LLM_TEMPERATURE={params['temperature']:.4f}\n")
                f.write(f"OLLAMA_NUM_THREAD={params['num_thread']}\n")
                f.write(f"OLLAMA_REPEAT_PENALTY={params['repeat_penalty']:.4f}\n")
                f.write(f"OLLAMA_TOP_P={params['top_p']:.4f}\n")

            logger.info(f"Successfully saved. Next run will automatically use these settings.")

        except IOError as e:
            logger.error(f"Failed to save optimized config: {e}")
