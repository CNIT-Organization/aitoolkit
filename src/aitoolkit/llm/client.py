"""LLM client backed by any OpenAI-compatible server (e.g. vLLM).

Provides three primitives, all provider-agnostic:

* :meth:`LLMClient.chat`             — single completion, returns text
* :meth:`LLMClient.stream`           — async token stream
* :meth:`LLMClient.chat_structured`  — completion validated into a Pydantic model

The client returns plain strings / Pydantic models and never leaks the underlying
``openai`` SDK types to callers.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import AsyncIterator, List, Optional, Type, TypeVar

from loguru import logger
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError

from aitoolkit.config import AIToolkitSettings, get_settings
from aitoolkit.exceptions import LLMError
from aitoolkit.types import ChatMessage, as_messages

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """A thin, stable wrapper over an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        settings: Optional[AIToolkitSettings] = None,
    ) -> None:
        settings = settings or get_settings()
        self.model = model or settings.llm_model
        self.default_temperature = (
            temperature if temperature is not None else settings.llm_temperature
        )
        self._base_url = base_url or settings.llm_base_url
        self._api_key = api_key or settings.llm_api_key
        self._timeout = timeout if timeout is not None else settings.llm_timeout
        self._max_retries = (
            max_retries if max_retries is not None else settings.llm_max_retries
        )

        self._aclient = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        self._sclient: Optional[OpenAI] = None  # created lazily for sync calls
        logger.info(f"LLMClient ready (model={self.model}, base_url={self._base_url})")

    @property
    def sync_client(self) -> OpenAI:
        """Lazily-created synchronous client (for non-async call sites)."""
        if self._sclient is None:
            self._sclient = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._sclient

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        prompt: Optional[str] = None,
        *,
        system: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Return a single completion as text."""
        msgs = as_messages(prompt, system=system, messages=messages)
        try:
            resp = await self._aclient.chat.completions.create(
                model=self.model,
                messages=msgs,  # type: ignore[arg-type]
                temperature=self._temp(temperature),
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - surface as toolkit error
            raise LLMError(f"chat completion failed: {exc}") from exc

        return resp.choices[0].message.content or ""

    def chat_sync(
        self,
        prompt: Optional[str] = None,
        *,
        system: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Synchronous counterpart of :meth:`chat`."""
        msgs = as_messages(prompt, system=system, messages=messages)
        try:
            resp = self.sync_client.chat.completions.create(
                model=self.model,
                messages=msgs,  # type: ignore[arg-type]
                temperature=self._temp(temperature),
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"chat completion failed: {exc}") from exc
        return resp.choices[0].message.content or ""

    # ---------------------------------------------------------------- stream
    async def stream(
        self,
        prompt: Optional[str] = None,
        *,
        system: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Yield completion text deltas as they arrive."""
        msgs = as_messages(prompt, system=system, messages=messages)
        try:
            stream = await self._aclient.chat.completions.create(
                model=self.model,
                messages=msgs,  # type: ignore[arg-type]
                temperature=self._temp(temperature),
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"streaming completion failed: {exc}") from exc

    # ------------------------------------------------------------ structured
    async def chat_structured(
        self,
        response_model: Type[T],
        prompt: Optional[str] = None,
        *,
        system: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None,
        temperature: Optional[float] = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """Return a completion validated into ``response_model``.

        Uses the OpenAI ``response_format`` json_schema mechanism, which vLLM
        implements via guided decoding. The raw JSON is validated with Pydantic,
        so a malformed response raises :class:`LLMError` rather than passing
        silently.
        """
        msgs = as_messages(prompt, system=system, messages=messages)
        schema = response_model.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": strict,
            },
        }
        try:
            resp = await self._aclient.chat.completions.create(
                model=self.model,
                messages=msgs,  # type: ignore[arg-type]
                temperature=self._temp(temperature),
                response_format=response_format,  # type: ignore[arg-type]
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"structured completion request failed: {exc}") from exc

        content = resp.choices[0].message.content or ""
        try:
            return response_model.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise LLMError(
                f"structured output did not match {response_model.__name__}: {exc}\n"
                f"raw content: {content[:500]}"
            ) from exc

    # ----------------------------------------------------------------- utils
    def _temp(self, temperature: Optional[float]) -> float:
        return temperature if temperature is not None else self.default_temperature

    async def aclose(self) -> None:
        await self._aclient.close()
        if self._sclient is not None:
            self._sclient.close()


@lru_cache(maxsize=8)
def _cached_client(model: Optional[str], temperature: Optional[float]) -> LLMClient:
    return LLMClient(model=model, temperature=temperature)


def get_llm_client(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> LLMClient:
    """Return a cached :class:`LLMClient` for the given model/temperature.

    Caching mirrors the previous ``llm.py`` behaviour and avoids re-creating
    HTTP clients on every request.
    """
    return _cached_client(model, temperature)
