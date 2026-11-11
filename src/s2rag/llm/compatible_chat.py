import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

import httpx
from filelock import FileLock
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)


class CacheOnlyMissError(RuntimeError):
    pass


def build_llm_cache_key(
    messages: Any,
    *,
    model: Any,
    seed: Any,
    temperature: Any,
    cache_namespace: str = "",
) -> str:
    key_data = {
        "messages": messages,
        "model": model,
        "seed": seed,
        "temperature": temperature,
    }
    if cache_namespace:
        key_data["cache_namespace"] = cache_namespace
    payload = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        namespace_override = kwargs.pop("_cache_namespace_override", None)
        messages = args[0] if args else kwargs.get("messages")
        if messages is None:
            raise ValueError("messages is required")

        defaults = self.llm_config.generate_params
        model = kwargs.get("model", defaults.get("model"))
        seed = kwargs.get("seed", defaults.get("seed"))
        temperature = kwargs.get("temperature", defaults.get("temperature"))
        namespace = (
            namespace_override
            if namespace_override is not None
            else str(getattr(self.global_config, "llm_cache_namespace", "") or "")
        )
        key = build_llm_cache_key(
            messages,
            model=model,
            seed=seed,
            temperature=temperature,
            cache_namespace=namespace,
        )

        lock_file = self.cache_file_name + ".lock"
        with FileLock(lock_file):
            connection = sqlite3.connect(self.cache_file_name)
            cursor = connection.cursor()
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, message TEXT, metadata TEXT)"
            )
            connection.commit()
            cursor.execute(
                "SELECT message, metadata FROM cache WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            connection.close()
        if row is not None:
            message, metadata_json = row
            metadata = json.loads(metadata_json)
            metadata.setdefault("request_cache_key", key)
            metadata.setdefault("request_id", "")
            return message, metadata, True

        if self.cache_only:
            raise CacheOnlyMissError(
                f"cache miss: key={key} cache={self.cache_file_name} "
                f"model={model} namespace={namespace!r}"
            )

        message, metadata = func(self, *args, **kwargs)
        metadata = dict(metadata or {})
        metadata.setdefault("request_cache_key", key)
        metadata.setdefault("request_id", "")
        with FileLock(lock_file):
            connection = sqlite3.connect(self.cache_file_name)
            cursor = connection.cursor()
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, message TEXT, metadata TEXT)"
            )
            cursor.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key, message, json.dumps(metadata)),
            )
            connection.commit()
            connection.close()
        return message, metadata, False

    return wrapper


def dynamic_retry(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        decorated = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_fixed(1),
            reraise=True,
        )(func)
        return decorated(self, *args, **kwargs)

    return wrapper


class CachedChatClient(BaseLLM):
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CachedChatClient":
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(
        self,
        cache_dir: str,
        global_config: BaseConfig,
        cache_filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config
        self.llm_name = global_config.llm_name
        self.llm_base_url = str(global_config.llm_base_url or "").rstrip("/")
        self.max_retries = int(kwargs.get("max_retries", global_config.max_retry_attempts))
        self.cache_only = bool(global_config.llm_cache_only)
        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)
        self._init_llm_config()
        self._client = None if self.cache_only else self._create_client()

    def _create_client(self) -> httpx.Client:
        if not self.llm_base_url:
            raise ValueError("llm_base_url is required for live requests")
        api_key = os.getenv("S2RAG_API_KEY", "").strip()
        if not api_key:
            raise ValueError("S2RAG_API_KEY is required for live requests")
        return httpx.Client(
            base_url=self.llm_base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
            timeout=httpx.Timeout(300.0, read=300.0),
        )

    def reset_connection_pool(self) -> None:
        if self.cache_only:
            return
        if self._client is not None:
            self._client.close()
        self._client = self._create_client()

    def _init_llm_config(self) -> None:
        config = dict(self.global_config.__dict__)
        config["generate_params"] = {
            "model": self.llm_name,
            "max_completion_tokens": config.get("max_new_tokens", 400),
            "n": config.get("num_gen_choices", 1),
            "seed": config.get("seed", 0),
            "temperature": config.get("temperature", 0.0),
        }
        self.llm_config = LLMConfig.from_dict(config_dict=config)

    def cache_key(self, messages: Any, **kwargs: Any) -> str:
        defaults = self.llm_config.generate_params
        return build_llm_cache_key(
            messages,
            model=kwargs.get("model", defaults.get("model")),
            seed=kwargs.get("seed", defaults.get("seed")),
            temperature=kwargs.get("temperature", defaults.get("temperature")),
            cache_namespace=str(self.global_config.llm_cache_namespace or ""),
        )

    def inspect_cache_coverage(self, messages_batch: Sequence[Any]) -> Dict[str, Any]:
        keys = [self.cache_key(messages) for messages in messages_batch]
        present = set()
        with FileLock(self.cache_file_name + ".lock"):
            if os.path.exists(self.cache_file_name):
                connection = sqlite3.connect(self.cache_file_name)
                try:
                    cursor = connection.cursor()
                    for start in range(0, len(keys), 800):
                        chunk = keys[start:start + 800]
                        placeholders = ",".join("?" for _ in chunk)
                        cursor.execute(
                            f"SELECT key FROM cache WHERE key IN ({placeholders})",
                            chunk,
                        )
                        present.update(str(row[0]) for row in cursor.fetchall())
                except sqlite3.OperationalError as exc:
                    if "no such table" not in str(exc).lower():
                        raise
                finally:
                    connection.close()
        records = [
            {"index": index, "key": key, "present": key in present}
            for index, key in enumerate(keys)
        ]
        missing = [record for record in records if not record["present"]]
        return {
            "cache_file": os.path.abspath(self.cache_file_name),
            "total": len(records),
            "present": len(records) - len(missing),
            "missing": len(missing),
            "complete": not missing,
            "records": records,
        }

    @cache_response
    @dynamic_retry
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        if self._client is None:
            raise RuntimeError("live client is unavailable in cache-only mode")
        params = deepcopy(self.llm_config.generate_params)
        params.update(kwargs)
        params["messages"] = messages
        if "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")
        response = self._client.post("/chat/completions", json=params)
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        message = choice["message"]["content"]
        usage = payload.get("usage") or {}
        metadata = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "finish_reason": str(choice.get("finish_reason") or ""),
            "request_id": str(payload.get("id") or ""),
        }
        return str(message), metadata
