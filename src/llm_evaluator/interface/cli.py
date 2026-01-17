import sys
import argparse
import asyncio
import logging
from pathlib import Path

from src.llm_evaluator.config import settings
from src.llm_evaluator.service.pipeline import EvaluationPipeline
from src.llm_evaluator.infrastructure.tuner import AutoTuner
from src.llm_evaluator.infrastructure.storage import CSVReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("llm_evaluator.cli")

async def _run_tuner(input_path: str) -> None:
    reader = CSVReader()
    all_tasks = list(reader.stream_input(input_path))

    if not all_tasks:
        logger.error("No tasks found in input file")
        sys.exit(1)

    tuner = AutoTuner(all_tasks, sample_size=10, n_trials=15)

    best_params = await tuner.optimize()
    tuner.save_recommendations(best_params)

async def _run_pipeline(input_path: str, output_path: str) -> None:
    pipeline = EvaluationPipeline()
    if not Path(input_path).exists():
        logger.critical(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        await pipeline.run(input_file=input_path, output_file=output_path)
    except Exception as e:
        logger.critical(f"Fatal pipeline error: {e}", exc_info=True)
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-Performance LLM Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    eval_parser = subparsers.add_parser("eval", help="Run evaluation pipeline")
    eval_parser.add_argument("input_file", type=str)
    eval_parser.add_argument("-o", "--output", type=str, default=settings.DEFAULT_OUTPUT_FILE)

    tune_parser = subparsers.add_parser("tune", help="Run auto-tuner")
    tune_parser.add_argument("input_file", type=str)

    args = parser.parse_args()

    logger.info(f"Initializing Pipeline with Model: {settings.MODEL_NAME}")
    logger.info(f"Concurrency limit: {settings.CONCURRENCY_LIMIT}")

    try:
        if args.command == "tune":
            asyncio.run(_run_tuner(args.input_file))
        else:
            input_f = getattr(args, "input_file", None)
            output_f = getattr(args, "output_file", None)
            if input_f:
                asyncio.run(_run_pipeline(input_f, output_f))
            else:
                parser.print_help()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting...")
        sys.exit(130)

if __name__ == "__main__":
    main()
