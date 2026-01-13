import sys
import argparse
import asyncio
import logging
from pathlib import Path

from src.llm_evaluator.config import settings
from src.llm_evaluator.service.pipeline import EvaluationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("llm_evaluator.cli")

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

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file containing questions"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        type=str,
        default=settings.DEFAULT_OUTPUT_FILE,
        help="Path where the result CSV will be written"
    )

    args = parser.parse_args()

    logger.info(f"Initializing Pipeline with Model: {settings.MODEL_NAME}")
    logger.info(f"Concurrency limit: {settings.CONCURRENCY_LIMIT}")

    try:
        asyncio.run(_run_pipeline(args.input_file, args.output_file))
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting...")
        sys.exit(130)

if __name__ == "__main__":
    main()
