# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

# @Date    : 2025-03-31
# @Author  : Zhaoyang & didi
# @Desc    :
import contextvars
import json
import os
import re
import threading
import time
from datetime import datetime
import yaml

from openai import AsyncOpenAI

from pathlib import Path
from typing import Dict, Optional, Any
from base.engine.cost_monitor import record_cost
from base.engine.logs import logger, LogLevel


# ---------------------------------------------------------------------------
# per-LLM-call structured event log.
# Writes one JSON line per LLM call to a shared events.jsonl in the run's
# result folder. A normal threading.Lock serializes append writes without
# introducing an await point after provider usage has been received. This keeps
# completed LLM calls from losing their event if the task is cancelled at the
# wall-clock cap. task_id propagates via contextvar set by the per-task runner.
# Set _events_path = None to disable (default).
# ---------------------------------------------------------------------------
_events_path: Optional[Path] = None
_events_lock: Optional[threading.Lock] = None
current_task_id: contextvars.ContextVar = contextvars.ContextVar("task_id", default="unknown")

# per-task LLM-call latency accumulator. Runner initializes a list
# in its context; AsyncLLM appends each call's latency_ms after the API
# returns. Sum at task end = active_llm_time_seconds (vs wall_clock_seconds).
current_task_latencies_ms: contextvars.ContextVar = contextvars.ContextVar(
    "task_latencies_ms", default=None
)


def set_events_path(p: Optional[Path]) -> None:
    """Set (or clear with None) the path to the per-run events.jsonl."""
    global _events_path, _events_lock
    _events_path = Path(p) if p is not None else None
    _events_lock = threading.Lock() if _events_path is not None else None


def _emit_event_sync(event: Dict[str, Any]) -> None:
    if _events_path is None or _events_lock is None:
        return
    with _events_lock:
        try:
            _events_path.parent.mkdir(parents=True, exist_ok=True)
            with _events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            logger.warning(f"[events.jsonl] write failed: {e}")


async def _emit_event(event: Dict[str, Any]) -> None:
    """Compatibility wrapper; production call sites should use sync emission.

    The synchronous helper avoids an await point between receiving provider
    usage and appending the event. Keep this wrapper only for any external tests
    or tools that imported the old async helper.
    """
    _emit_event_sync(event)

class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.top_p = config.get("top_p", 1)

class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""
    
    _instance = None  # For singleton pattern if needed
    _default_config = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with an optional configuration dictionary"""
        self.configs = config_dict or {}
    
    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            config_data: Optional[Dict[str, Any]] = None

            project_root = Path(__file__).resolve().parents[2]
            relative_config_paths = [
                Path("config/global_config.yaml"),
                Path("config/global_config2.yaml"),
                Path("config/model_config.yaml"),
            ]
            config_paths = (
                [project_root / path for path in relative_config_paths]
                + relative_config_paths
            )

            config_file = next((path for path in config_paths if path.exists() and path.stat().st_size > 0), None)

            if config_file is not None:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = cls._load_config_from_env()

            if not config_data:
                raise FileNotFoundError(
                    "No default configuration file found in the expected locations and no environment-based fallback is configured."
                )

            if "models" in config_data:
                config_data = config_data["models"] or {}

            cls._default_config = cls(config_data)

        return cls._default_config

    @classmethod
    def _load_config_from_env(cls) -> Optional[Dict[str, Any]]:
        """Build configuration from environment variables when no YAML file is present."""
        inline_config = os.getenv("AUTOENV_MODEL_CONFIG_JSON")
        if inline_config:
            try:
                data = yaml.safe_load(inline_config)
            except yaml.YAMLError:
                logger.log_to_file(
                    LogLevel.WARNING,
                    "Failed to parse AUTOENV_MODEL_CONFIG_JSON; falling back to explicit env vars.",
                )
            else:
                if isinstance(data, dict):
                    return data.get("models", data) or {}

        api_key = os.getenv("AUTOENV_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        base_url = (
            os.getenv("AUTOENV_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )

        def _get_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                logger.log_to_file(
                    LogLevel.WARNING,
                    f"Invalid float value for {name}: {raw!r}; using {default}.",
                )
                return default

        temperature = _get_float("AUTOENV_OPENAI_TEMPERATURE", 1)
        top_p = _get_float("AUTOENV_OPENAI_TOP_P", 1)

        models_env = os.getenv("AUTOENV_OPENAI_MODELS", "o3")
        models = [m.strip() for m in models_env.split(",") if m.strip()]

        if not models:
            models = ["o3"]

        config: Dict[str, Any] = {}
        for model_name in models:
            normalized = model_name.upper().replace('-','_').replace('/','_')
            env_key_name = f"AUTOENV_{normalized}_API_KEY"
            env_base_name = f"AUTOENV_{normalized}_BASE_URL"

            model_api_key = os.getenv(env_key_name, api_key)
            model_base_url = os.getenv(env_base_name, base_url)

            config[model_name] = {
                "api_key": model_api_key,
                "base_url": model_base_url,
                "temperature": temperature,
                "top_p": top_p,
            }

        return config
    
    def get(self, llm_name: str) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")
        
        config = self.configs[llm_name]
        base_url = config.get("base_url", "https://api.openai.com/v1")
        api_key = (
            self._api_key_from_env(llm_name, base_url, include_generic=False)
            or config.get("api_key")
            or config.get("key")
            or self._api_key_from_env(llm_name, base_url)
        )
        
        # Create a config dictionary with the expected keys for LLMConfig
        llm_config = {
            "model": llm_name,  # Use the key as the model name
            "temperature": config.get("temperature", 1),
            "key": api_key,  # Map api_key to key
            "base_url": base_url,
            "top_p": config.get("top_p", 1)  # Add top_p parameter
        }
        
        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)

    @staticmethod
    def _api_key_from_env(
        llm_name: str,
        base_url: str,
        include_generic: bool = True,
    ) -> Optional[str]:
        """Return a provider-appropriate API key from the environment, if present."""
        normalized = llm_name.upper().replace("-", "_").replace("/", "_").replace(".", "_")
        candidates = [
            f"AUTOENV_{normalized}_API_KEY",
        ]

        base = (base_url or "").lower()
        if "openrouter" in base:
            candidates.append("OPENROUTER_API_KEY")
        if "generativelanguage.googleapis.com" in base:
            candidates.extend(["GEMINI_API_KEY", "GOOGLE_API_KEY"])
        if "deepseek" in base:
            candidates.append("DEEPSEEK_API_KEY")

        if include_generic:
            candidates.extend(["AUTOENV_OPENAI_API_KEY", "OPENAI_API_KEY"])
        for name in candidates:
            value = os.getenv(name)
            if value:
                return value
        return None
    
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = config
    
    def get_all_names(self) -> list:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())
    
class ModelPricing:
    """Pricing information for different models in USD per 1K tokens"""
    PRICES = {
        "gemini-3-flash-preview": {"input": 0.0005, "output": 0.003},
    }


    # Models we have already warned about (avoid log spam — warn once each).
    _unpriced_warned: set = set()

    @classmethod
    def get_price(cls, model_name, token_type):
        """Get the price per 1K tokens for a specific model and token type (input/output)"""
        # Try to find exact match first
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]

        matches = [key for key in cls.PRICES if key in model_name]
        if matches:
            best = max(matches, key=len)
            return cls.PRICES[best][token_type]

        if model_name not in cls._unpriced_warned:
            cls._unpriced_warned.add(model_name)
            logger.warning(
                f"[ModelPricing] No price-table entry for model '{model_name}'. "
                f"Price-table cost estimate will be $0 for this model. This is only "
                f"correct if your provider returns an authoritative usage.cost "
                f"(e.g. OpenRouter with extra_body usage.include=True); otherwise "
                f"add '{model_name}' to ModelPricing.PRICES or reported cost will be "
                f"under-counted."
            )
        return 0

class TokenUsageTracker:
    """Tracks token usage and calculates costs"""
    def __init__(self, model: str = ""):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []
    
    def add_usage(self, model, input_tokens, output_tokens, actual_cost=None):
        """Add token usage for a specific API call.

        actual_cost: the provider-reported billed cost (USD), e.g. OpenRouter's
        usage.cost. When present (including 0.0) it is used as the authoritative total
        cost instead of the static price-table estimate, which can be stale or
        mis-keyed. The per-token input/output split is still recorded from the
        table for reference.
        """
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        if actual_cost is not None and actual_cost >= 0:
            total_cost = actual_cost
            cost_source = "openrouter_actual"
        else:
            total_cost = input_cost + output_cost
            cost_source = "price_table"

        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_source": cost_source,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output")
            }
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self):
        """Get a summary of token usage and costs"""
        return {
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }

class AsyncLLM:
    def __init__(self, config, system_msg:str = None, max_completion_tokens:int = None, role: str = "unknown"):
        """
        Initialize the AsyncLLM with a configuration

        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
            max_tokens: Optional maximum number of tokens to generate
            role: Logical role for the events.jsonl log (e.g. "PlannerAgent",
                  "ImplementerAgent", "summarizer", "verifier"). Default "unknown".
        """
        # Handle the case where config is a string (LLM name)
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)

        # At this point, config should be an LLMConfig instance
        self.config = config
        self.aclient = AsyncOpenAI(
            api_key=self.config.key,
            base_url=self.config.base_url,
            max_retries=8,
        )
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker(model=self.config.model)
        self.max_completion_tokens = max_completion_tokens
        self.role = role
        self.prompt_cache_enabled = False
        self.prompt_cache_min_prefix_chars = 16000

    def _supports_explicit_prompt_cache(self) -> bool:
        base_url = getattr(self.config, "base_url", "") or ""
        model = (getattr(self.config, "model", "") or "").lower()
        return "openrouter" in base_url.lower() and "claude" in model

    def _split_cacheable_prompt(self, prompt: str) -> tuple[str, str] | None:
        """Split prompt into stable prefix + dynamic suffix for cache_control.

        The prompt builders keep changing task state/memory/observations late in
        the prompt. Marking the whole prompt would create a new cache entry every
        call and usually lose money. Only add cache_control when a known runtime
        marker lets us isolate a sufficiently large static prefix.
        """
        if not isinstance(prompt, str):
            return None
        for marker in ("\n==== Runtime State ====\n", "\n==== Progress ====\n"):
            idx = prompt.find(marker)
            if idx > 0:
                prefix = prompt[:idx]
                suffix = prompt[idx:]
                if len(prefix) >= int(self.prompt_cache_min_prefix_chars or 0):
                    return prefix, suffix
        return None

    def _build_user_content(self, prompt):
        if not (
            self.prompt_cache_enabled
            and isinstance(prompt, str)
            and self._supports_explicit_prompt_cache()
        ):
            return prompt

        split = self._split_cacheable_prompt(prompt)
        if not split:
            return prompt

        prefix, suffix = split
        return [
            {
                "type": "text",
                "text": prefix,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": suffix,
            },
        ]
        
    async def __call__(self, prompt, max_tokens=None, role_override: Optional[str] = None):
        """Send a prompt to the LLM. Accepts text (str) or multimodal payloads (list).

        `role_override` swaps the per-call `role` tag in events.jsonl without
        spinning up a second AsyncLLM instance. Used so memory-compression
        ("Summarizer") cost is no longer hidden under whatever role the
        underlying worker LLM carries (`ImplementerAgent`, `PlannerAgent`,
        ...). Defaults to `self.role`.
        """
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })

        # Support plain text prompts and multimodal payloads (list)
        if isinstance(prompt, str):
            message.append({"role": "user", "content": self._build_user_content(prompt)})
        elif isinstance(prompt, list):
            message.append({"role": "user", "content": prompt})
        else:
            raise ValueError(f"prompt must be str or list, got {type(prompt)}")

        # Prefer to use the max_tokens argument passed to the function; if it is None, use the instance variable.
        tokens_to_use = max_tokens if max_tokens is not None else self.max_completion_tokens

        # Claude models via Bedrock don't support both temperature and top_p
        is_claude = "claude" in self.config.model.lower()
        sampling_params = (
            {"temperature": self.config.temperature}
            if is_claude
            else {"temperature": self.config.temperature, "top_p": self.config.top_p}
        )
        # Time the actual API round-trip for the events.jsonl latency_ms field.
        t0 = time.perf_counter()

        extra_kwargs = (
            {"extra_body": {"usage": {"include": True}}}
            if "openrouter" in (getattr(self.config, "base_url", "") or "")
            else {}
        )
        if tokens_to_use is not None and "o3" in self.config.model:
            response = await self.aclient.chat.completions.create(
                model=self.config.model,
                messages=message,
                max_completion_tokens=tokens_to_use,
                **sampling_params,
                **extra_kwargs,
            )
        # Only gpt-series support max_completion_tokens.
        elif tokens_to_use is not None and "o3" not in self.config.model:
            response = await self.aclient.chat.completions.create(
                model=self.config.model,
                messages=message,
                max_tokens=tokens_to_use,
                **sampling_params,
                **extra_kwargs,
            )
        else:
            response = await self.aclient.chat.completions.create(
                model=self.config.model,
                messages=message,
                **sampling_params,
                **extra_kwargs,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Extract token usage from response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        actual_cost = getattr(response.usage, "cost", None)
        if actual_cost is None:
            extra = getattr(response.usage, "model_extra", None) or {}
            actual_cost = extra.get("cost")
        try:
            actual_cost = float(actual_cost) if actual_cost is not None else None
        except (TypeError, ValueError):
            actual_cost = None

        def _usage_value(name: str, default=0):
            value = getattr(response.usage, name, None)
            if value is None:
                extra = getattr(response.usage, "model_extra", None) or {}
                value = extra.get(name)
            try:
                return int(value or default)
            except (TypeError, ValueError):
                return default

        cache_read_tokens = _usage_value("cache_read_input_tokens")
        cache_write_tokens = _usage_value("cache_creation_input_tokens")
        if cache_read_tokens == 0:
            cache_read_tokens = _usage_value("cached_tokens")
        if cache_write_tokens == 0:
            cache_write_tokens = _usage_value("cache_write_tokens")

        # Track token usage and calculate cost
        usage_record = self.usage_tracker.add_usage(
            self.config.model,
            input_tokens,
            output_tokens,
            actual_cost=actual_cost,
        )

        # Report to global cost monitor if active
        record_cost(self.config.model, input_tokens, output_tokens, usage_record["total_cost"])

        event = {
            "ts": datetime.now().isoformat(),
            "task_id": current_task_id.get(),
            "role": role_override or self.role,
            "model": self.config.model,
            "prompt_tokens": int(input_tokens),
            "completion_tokens": int(output_tokens),
            "cost": round(float(usage_record["total_cost"]), 6),
            "cost_source": usage_record.get("cost_source"),
            "latency_ms": round(latency_ms, 2),
        }
        if cache_read_tokens or cache_write_tokens:
            event["cache_read_tokens"] = cache_read_tokens
            event["cache_write_tokens"] = cache_write_tokens
        _emit_event_sync(event)

        # accumulate per-task active LLM latency.
        latencies = current_task_latencies_ms.get()
        if latencies is not None:
            latencies.append(latency_ms)

        ret = response.choices[0].message.content
        if ret is None:
            choice0 = response.choices[0]
            finish_reason = getattr(choice0, "finish_reason", "?")
            rc = getattr(choice0.message, "reasoning_content", None)
            rc_chars = len(rc) if isinstance(rc, str) else 0
            logger.warning(
                f"[AsyncLLM] message.content=None model={self.config.model} "
                f"finish_reason={finish_reason} reasoning_content_chars={rc_chars} "
                f"role={self.role} task_id={current_task_id.get()}"
            )
            ret = ""
        logger.log_to_file(LogLevel.INFO, f"LLM Response: {ret}")

        return ret
    
    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()

    async def generate_text_to_image(self, prompt: str) -> dict[str, Any]:
        """
        Text-to-image generation.

        Args:
            prompt: Text prompt for image generation

        Returns:
            {
                'success': bool,
                'image_base64': str | None,
                'prompt': str,
                'error': str | None
            }
        """
        try:
            response = await self(prompt)
            image_b64 = self._extract_image_from_response(response)

            if not image_b64:
                return {
                    "success": False,
                    "image_base64": None,
                    "prompt": prompt,
                    "error": "No image found in response",
                }

            return {
                "success": True,
                "image_base64": image_b64,
                "prompt": prompt,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "image_base64": None,
                "prompt": prompt,
                "error": str(e),
            }

    async def generate_image_to_image(
        self, prompt: str, reference_images: list[str]
    ) -> dict[str, Any]:
        """
        Image-to-image generation with style references.

        Args:
            prompt: Text prompt for image generation
            reference_images: List of base64-encoded reference images

        Returns:
            {
                'success': bool,
                'image_base64': str | None,
                'prompt': str,
                'error': str | None
            }
        """
        try:
            content = []
            for img_b64 in reference_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )
            content.append({"type": "text", "text": prompt})

            response = await self(content)
            image_b64 = self._extract_image_from_response(response)

            if not image_b64:
                return {
                    "success": False,
                    "image_base64": None,
                    "prompt": prompt,
                    "error": "No image found in response",
                }

            return {
                "success": True,
                "image_base64": image_b64,
                "prompt": prompt,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "image_base64": None,
                "prompt": prompt,
                "error": str(e),
            }

    def _extract_image_from_response(self, response: str) -> str | None:
        """Extract base64 image from LLM response."""
        if not isinstance(response, str):
            return None
        match = re.search(r"data:image/[^;]+;base64,([^)]+)", response)
        return match.group(1) if match else None


def create_llm_instance(llm_config, role: str = "unknown") -> AsyncLLM:
    """
    Create an AsyncLLM instance using the provided configuration

    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
        role: Logical role for events.jsonl tagging ("PlannerAgent",
              "ImplementerAgent", "summarizer", etc.). Forwarded to AsyncLLM.

    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config, role=role)

    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config, role=role)  # AsyncLLM constructor handles lookup

    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config, role=role)

    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")
