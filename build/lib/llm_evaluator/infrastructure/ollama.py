import time
import asyncio
import httpx
import logging
from typing import Optional, Self, Any

from src.llm_evaluator.config import settings, Settings
from src.llm_evaluator.domain.models import LLMResponse
from src.llm_evaluator.domain.exceptions import LLMConnectionError

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, config: Settings = settings) -> None:
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.MODEL_NAME
        self.timeout = settings.HTTP_TIMEOUT_SECONDS
        self.default_options = {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": config.LLM_MAX_TOKENS,
            "num_ctx": 4096,
            "repeat_penalty": 1.1,
            "stop": ["<|endoftext|>", "User:", "ANSWER:"]
        }
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def generate(self, prompt: str, options_override: Optional[dict]) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("OllamaClient must be used within an 'async with' block.")

        url = f"{self.base_url}/api/generate"
        final_options = self.default_options.copy()
        if options_override:
            final_options.update(options_override)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": final_options
        }

        max_retries = 3
        for attempt in range(max_retries):
            start_time = time.perf_counter()
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                latency_ms = (time.perf_counter() - start_time) * 1000

                return LLMResponse(
                    raw_content=data.get("response", ""),
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    total_duration_ns=data.get("total_duration", 0),
                    client_latency_ms=latency_ms
                )
            except (httpx.ReadTimeout, httpx.ConnectError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                if attempt == max_retries - 1:
                    raise LLMConnectionError(f"Failed after {max_retries} attempts: {e}")
                await asyncio.sleep(5)
            except httpx.HTTPStatusError as e:
                raise LLMConnectionError(f"Ollama API Error: {e.response.status_code}")
