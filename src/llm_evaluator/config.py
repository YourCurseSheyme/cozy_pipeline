import os
import logging
from typing import Dict, Tuple, List, Union
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = _CURRENT_FILE.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("llm_evaluator.config")

_env_files: List[Union[str, Path]] = [PROJECT_ROOT / ".env"]
_optimized_path = PROJECT_ROOT / ".env.optimized"

if _optimized_path.exists():
    _env_files.append(_optimized_path)
    logger.info(f"Optimized config detected. Loading overrides from {_optimized_path}")
else:
    logger.info(f"No optimized config detected. Loading default config")

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "qwen2.5:7b-instruct"

    CONCURRENCY_LIMIT: int = 1
    HTTP_TIMEOUT_SECONDS: float = 600.0

    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

    OLLAMA_NUM_CTX: int = 4096
    OLLAMA_NUM_THREAD: int = 4
    OLLAMA_REPEAT_PENALTY: float = 1.1
    OLLAMA_TOP_P: float = 1.0

    DEFAULT_INPUT_FILE: str =  str(DATA_DIR / "input.csv")
    DEFAULT_OUTPUT_FILE: str = str(DATA_DIR / "results.csv")

    KERNEL_BOOTLOADER: str = ("""
        ROLE
        You are a reasoning agent whose sole responsibility is to select exactly one correct option index from the given list.

        TASK SPACE
        You are given:
        - a question,
        - a fixed list of answer options indexed from 0.
        The option set is complete and immutable.

        CORRECTNESS CRITERIA
        An answer is correct if and only if it:
        - does not contradict established facts or standard definitions of the domain,
        - is logically consistent with the question,
        - best matches the intended meaning of the question.

        EPISTEMIC STANDARDS (THE TRUTH PROTOCOL)
        - **Priority of Definition:** Established academic knowledge and standard definitions override option phrasing.
        - **Silence is not Truth:** "Not specified" or "Unknown" is NOT equivalent to "True".
        - **No Hallucination:** Do not invent missing premises, entities, or motivations to justify an option.
        - **Strict Falsification:** If a statement asserts a specific fact (mathematical, historical, legal) without defining context, and it contradicts standard theory, it is FALSE.
        - **Non-Derivability:** A statement that does not follow from standard definitions is FALSE, even if not self-contradictory.

        DECISION HEURISTICS
        - **The Counter-Example Rule:** A universal claim ("always", "all", "never") is INVALIDated by a single counter-example.
        - **Occam's Razor (Minimal Claim):** When multiple options are plausible, select the one that makes the weakest necessary claim while remaining correct.
        - **Fact Anchoring:** Treat options introducing new entities/conditions not present in the prompt with extreme suspicion.
        - **Structural Rigor:** If a statement asserts a concrete structural property (e.g., "X is an ideal") but violates the formal definition, it is FALSE.

        VALIDATION CHECK
        Before finalizing, ensure the selected option is internally coherent and contradicts neither the question nor the domain axioms.

        OUTPUT CONTRACT
        1. You may generate a concise reasoning block to verify your logic.
        2. The very last line of your output MUST be exactly:
        ANSWER: <index>
    """)

    KERNEL_DRIVERS: Dict[str, str] = {
        "math": ("""
            MATH DOMAIN CONTEXT
            - **Definitions are Absolute:** Standard definitions (e.g., Bourbaki) apply unless explicitly redefined in the text.
            - **Logic of Falsification:** A universal claim ("For all X...") is invalidated by a SINGLE counter-example.
            - **Structure Verification:** For algebraic structures (Group, Ring, Ideal), ALL axioms must hold. Missing one = False.
            - **Implication Logic:** "A implies B" is False only if A is True and B is False.
            - **Subset Warning:** Not every subset closed under addition is an ideal (check absorption).
    """),

        "computer science": ("""
            COMPUTER SCIENCE DOMAIN CONTEXT
            - **Execution Semantics:** Treat the question as deterministic code evaluation, not natural language.
            - **ASCII Rigor:** String comparison is strictly numeric based on ASCII values ('Z' < 'a').
            - **Lexicographical Rule:** Comparison stops at the first differing character. Length is irrelevant unless prefixes match.
            - **Boolean Logic:** Evaluate expressions fully. Short-circuit logic applies only if explicitly supported.
    """),

        "history": ("""
            HISTORY DOMAIN CONTEXT
            - **Chronology Rule:** Anachronisms (events/people out of time) immediately invalidate an option.
            - **Consensus:** Prefer mainstream academic consensus over fringe theories.
            - **Contextual Truth:** Interpret statements within the specific era mentioned. Avoid "presentism".
    """),

        "law": ("""
            LAW DOMAIN CONTEXT
            - **Hierarchy of Authority:** Constitution > Statute > Regulation > Case Law.
            - **Elements Test:** A legal claim is valid only if ALL constituent elements of the rule are satisfied.
            - **Literalism:** Apply statutes exactly as written. Do not infer "spirit of the law" unless asked.
            - **Distinction:** Ensure cited cases are factually analogous; distinguish based on material facts.
    """),

        "default": ("""
            GENERAL REASONING CONTEXT
            - **Semantic Consistency:** The correct option must preserve the core meaning of the premise without distortion.
            - **Information Sufficiency:** Reject options that require information not present in the context (unless it is common knowledge).
            - **Logical Consistency:** Eliminate options that contain internal contradictions.

            LEGAL INTERPRETATION CANON
            - Prefer established judicial interpretations over abstract balancing arguments.
            - If an option reflects a well-known Supreme Court holding or classroom-standard formulation,
              prefer it over generalized philosophical statements.
            - When multiple options seem reasonable, select the one most commonly cited in legal education.
    """)
    }

    model_config = SettingsConfigDict(
        env_file=tuple(_env_files),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True
    )

settings = Settings()
