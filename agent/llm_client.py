"""
LLM Client for ML Lab Agent.

Features:
- Multi-provider fallback chain: Groq -> NVIDIA NIM -> OpenRouter -> Ollama
- Multi-key rotation: Pools 3 Groq API keys with round-robin and rate-limit (HTTP 429) failover
- Reasoning / thought-tag handling (handles <think>, reasoning fields from gpt-oss/nemotron)
- Structured JSON output with Pydantic validation and automatic self-repair retry loop
- Full observability: latencies, token counts, and provider attribution
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Type, TypeVar
import httpx
from pydantic import BaseModel, ValidationError
import yaml

from agent.schemas import AgentAction, LLMProviderConfig, LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Robust, fault-tolerant LLM client with provider cascading and key pooling.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.providers: list[dict[str, Any]] = []
        self.key_indices: dict[str, int] = {}  # provider_name -> current key index
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load LLM providers and initialize key pools."""
        from dotenv import load_dotenv
        load_dotenv()

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        providers_raw = cfg.get("llm", {}).get("providers", [])
        for p in providers_raw:
            name = p.get("name", "")
            base_url = p.get("base_url", "")
            model = p.get("model", "")
            max_tokens = p.get("max_tokens", 2048)
            temperature = p.get("temperature", 0.3)
            timeout = p.get("timeout_seconds", 30)
            max_retries = p.get("max_retries", 2)

            # Discover all valid keys for this provider
            keys: list[str] = []
            
            # Check api_key_envs list first
            for env_var in p.get("api_key_envs", []):
                val = os.getenv(env_var)
                if val and val.strip() and val not in keys:
                    keys.append(val.strip())

            # Check single api_key_env
            single_env = p.get("api_key_env")
            if single_env:
                val = os.getenv(single_env)
                if val and val.strip() and val not in keys:
                    keys.append(val.strip())

            # Special case for groq if env has GROQ_API_KEY, GROQ_API_KEY2, GROQ_API_KEY3
            if name.lower() == "groq":
                for fallback_env in ["GROQ_API_KEY", "GROQ_API_KEY2", "GROQ_API_KEY3"]:
                    val = os.getenv(fallback_env)
                    if val and val.strip() and val not in keys:
                        keys.append(val.strip())

            # Ollama does not require keys
            if name.lower() == "ollama" or keys:
                self.providers.append({
                    "name": name,
                    "base_url": base_url,
                    "model": model,
                    "keys": keys,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "timeout": timeout,
                    "max_retries": max_retries,
                })
                self.key_indices[name] = 0

        provider_summary = [(p["name"], f"{len(p['keys'])} keys") for p in self.providers]
        logger.info(f"Initialized LLMClient with {len(self.providers)} providers: {provider_summary}")

    def _get_next_key(self, provider: dict[str, Any]) -> str | None:
        """Get the current API key for a provider and advance round-robin index."""
        keys = provider.get("keys", [])
        if not keys:
            return None
        name = provider["name"]
        idx = self.key_indices.get(name, 0) % len(keys)
        key = keys[idx]
        self.key_indices[name] = (idx + 1) % len(keys)
        return key

    def _call_single_provider(
        self,
        provider: dict[str, Any],
        messages: list[dict[str, str]],
        json_mode: bool = True,
    ) -> LLMResponse:
        """Execute request against a single provider with key rotation and retries."""
        name = provider["name"]
        base_url = provider["base_url"].rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        keys = provider.get("keys", [])
        model = provider["model"]
        max_tokens = provider["max_tokens"]
        temperature = provider["temperature"]
        timeout = float(provider["timeout"])
        max_retries = provider.get("max_retries", 2)

        # Total attempts allowed for this provider is based on number of keys + retries
        total_attempts = max(len(keys), 1) * (max_retries + 1)
        last_exception: Exception | None = None

        for attempt in range(total_attempts):
            api_key = self._get_next_key(provider)
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode and name in ["groq", "openai"]:
                payload["response_format"] = {"type": "json_object"}

            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(endpoint, headers=headers, json=payload)

                latency_ms = (time.perf_counter() - t0) * 1000

                # Check for rate limiting (429) or temporary server errors (500, 502, 503, 504)
                if resp.status_code == 429 or resp.status_code >= 500:
                    logger.warning(
                        f"Provider '{name}' returned {resp.status_code}. "
                        f"Rotating key/retrying (attempt {attempt + 1}/{total_attempts})."
                    )
                    time.sleep(0.5)
                    continue

                resp.raise_for_status()
                data = resp.json()

                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content") or ""

                # Handle reasoning fields (e.g. OpenAI gpt-oss or reasoning models)
                if not content.strip() and "reasoning" in message:
                    reasoning_text = message.get("reasoning", "")
                    content = reasoning_text

                usage = data.get("usage", {})
                return LLMResponse(
                    content=content,
                    provider_used=name,
                    model_used=model,
                    latency_ms=latency_ms,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error from {name}: {e.response.status_code} - {e.response.text[:150]}")
                last_exception = e
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error calling {name}: {e}")
                last_exception = e
                time.sleep(0.5)

        raise RuntimeError(f"All attempts for provider '{name}' failed. Last error: {last_exception}")

    def call(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
    ) -> LLMResponse:
        """
        Send chat completion request through fallback provider cascade:
        Groq (3 keys) -> NVIDIA NIM -> OpenRouter -> Ollama
        """
        errors = []
        for provider in self.providers:
            try:
                response = self._call_single_provider(provider, messages, json_mode=json_mode)
                if response.content and response.content.strip():
                    return response
            except Exception as e:
                logger.warning(f"Provider '{provider['name']}' cascade fallback: {e}")
                errors.append(f"{provider['name']}: {str(e)}")

        raise RuntimeError(f"All LLM providers failed in cascade: {'; '.join(errors)}")

    def call_structured(
        self,
        messages: list[dict[str, str]],
        schema_cls: Type[T],
        max_repair_attempts: int = 2,
    ) -> tuple[T, LLMResponse]:
        """
        Call LLM, extract JSON, validate against Pydantic schema,
        and auto-repair if validation fails.
        """
        current_messages = list(messages)

        for attempt in range(max_repair_attempts + 1):
            llm_response = self.call(current_messages, json_mode=True)
            raw_text = llm_response.content

            # Parse and validate JSON
            parsed_json = self.extract_json(raw_text)
            if parsed_json is not None:
                try:
                    validated = schema_cls.model_validate(parsed_json)
                    return validated, llm_response
                except ValidationError as ve:
                    err_msg = str(ve)
                    logger.warning(f"Schema validation error (attempt {attempt + 1}): {err_msg}")
                    if attempt < max_repair_attempts:
                        current_messages.append({"role": "assistant", "content": raw_text})
                        current_messages.append({
                            "role": "user",
                            "content": (
                                f"Your previous JSON output failed validation with the following error:\n"
                                f"{err_msg}\n\n"
                                f"Please fix the error and output ONLY the corrected JSON object matching the schema."
                            ),
                        })
                        continue
            else:
                logger.warning(f"Failed to parse JSON from response (attempt {attempt + 1}): {raw_text[:200]}")
                if attempt < max_repair_attempts:
                    current_messages.append({"role": "assistant", "content": raw_text})
                    current_messages.append({
                        "role": "user",
                        "content": (
                            "Your previous output was not valid JSON. "
                            "Please respond with ONLY a valid JSON object matching the required schema. "
                            "Do not include explanation outside the JSON."
                        ),
                    })
                    continue

        raise ValueError(
            f"Failed to produce valid structured output for {schema_cls.__name__} "
            f"after {max_repair_attempts + 1} attempts. Last response:\n{raw_text}"
        )

    @staticmethod
    def extract_json(text: str) -> dict[str, Any] | None:
        """
        Robustly extracts JSON from raw LLM output.
        Handles:
        - Markdown ```json ... ``` blocks
        - <think>...</think> tags
        - Leading/trailing conversational commentary
        """
        if not text:
            return None

        # Remove <think>...</think> reasoning blocks if present
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Try direct json parse first
        try:
            val = json.loads(clean_text)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

        # Look for markdown code fence
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, flags=re.DOTALL)
        if match:
            try:
                val = json.loads(match.group(1))
                if isinstance(val, dict):
                    return val
            except Exception:
                pass

        # Find first '{' and last '}'
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = clean_text[start : end + 1]
            try:
                val = json.loads(snippet)
                if isinstance(val, dict):
                    return val
            except Exception:
                pass

        return None
