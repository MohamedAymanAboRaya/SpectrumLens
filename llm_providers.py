"""
Unified LLM Provider — AgentRouter + OpenRouter + Groq with automatic fallback.

Usage:
    from llm_providers import LLMProvider
    provider = LLMProvider()
    text, source = provider.call(messages, role="scope")  # role: scope/critic/generate
    for token in provider.stream(messages, role="generate"):
        print(token, end="")
"""

import os
import json
import logging
import time
from typing import List, Dict, Optional, Tuple, Generator

logger = logging.getLogger(__name__)

# ─── API Keys (read at call time, not import time, so load_dotenv works) ──────
def _get_key(name: str) -> str:
    return os.environ.get(name, "")

def _get_base(name: str, default: str) -> str:
    return os.environ.get(name, default)

# ─── Model Defaults per Role ─────────────────────────────────────────────────
# scope: fast, structured JSON; critic: fast scoring; generate: best quality
MODEL_DEFAULTS = {
    "agentrouter": {
        "scope":    "gpt-5.6-sol",
        "critic":   "gpt-5.6-sol",
        "generate": "gpt-5.6-sol",
        "stream":   "gpt-5.6-sol",
    },
    "openrouter": {
        "scope":    "google/gemini-2.5-flash",
        "critic":   "google/gemini-2.5-flash",
        "generate": "google/gemini-2.5-flash",
        "stream":   "google/gemini-2.5-flash",
    },
    "groq": {
        "scope":    "allam-2-7b",
        "critic":   "allam-2-7b",
        "generate": "openai/gpt-oss-120b",
        "stream":   "openai/gpt-oss-120b",
    },
}

# Fallback order: AgentRouter → OpenRouter → Groq
PROVIDER_ORDER = ["agentrouter", "openrouter", "groq"]


def _strip_think(raw: str) -> str:
    """Remove <think>...</think> blocks from LLM output.
    IMPORTANT: No .strip() here — called on individual streaming tokens,
    stripping would eat the boundary spaces between words.
    """
    import re
    return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)


def _strip_think_full(raw: str) -> str:
    """Remove <think>...</think> blocks and strip outer whitespace.
    Safe for non-streaming full responses where the complete text is
    available at once — .strip() is fine here since we're not mid-stream.
    """
    import re
    return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()


class LLMProvider:
    """
    Unified LLM provider with automatic fallback across AgentRouter, OpenRouter, and Groq.
    
    Usage:
        provider = LLMProvider()
        text, source = provider.call(messages, role="generate")
        for token in provider.stream(messages, role="generate"):
            print(token, end="")
    """

    def __init__(self, preferred_provider: Optional[str] = None):
        """
        Args:
            preferred_provider: Force a specific provider ("agentrouter", "openrouter", "groq").
                               If None, tries all in fallback order.
        """
        self._preferred = preferred_provider
        self._last_provider = None
        self._last_model = None

    def _get_available_providers(self) -> List[str]:
        """Return ordered list of available providers."""
        available = []
        for p in PROVIDER_ORDER:
            if p == "agentrouter" and _get_key("AGENTROUTER_API_KEY"):
                available.append(p)
            elif p == "openrouter" and _get_key("OPENROUTER_API_KEY"):
                available.append(p)
            elif p == "groq" and _get_key("GROQ_API_KEY"):
                available.append(p)
        if self._preferred and self._preferred in available:
            available = [self._preferred] + [p for p in available if p != self._preferred]
        return available

    def _call_agentrouter(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> str:
        """Call AgentRouter API."""
        import requests
        resp = requests.post(
            f"{_get_base('AGENTROUTER_BASE_URL', 'https://agentrouter.org/v1')}/chat/completions",
            headers={
                "Authorization": f"Bearer {_get_key('AGENTROUTER_API_KEY')}",
                "Content-Type": "application/json",
                "User-Agent": "opencode/1.0",
                "HTTP-Referer": "https://opencode.ai",
            },
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_openrouter(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> str:
        """Call OpenRouter API."""
        import requests
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_get_key('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_groq(self, messages: List[Dict], model: str, temperature: float, max_tokens: int,
                   response_format=None) -> str:
        """Call Groq API."""
        from groq import Groq
        client = Groq(api_key=_get_key("GROQ_API_KEY"))
        kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if response_format and "allam" not in model:
            kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def _stream_agentrouter(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
        """Stream from AgentRouter."""
        import requests
        resp = requests.post(
            f"{_get_base('AGENTROUTER_BASE_URL', 'https://agentrouter.org/v1')}/chat/completions",
            headers={
                "Authorization": f"Bearer {_get_key('AGENTROUTER_API_KEY')}",
                "Content-Type": "application/json",
                "User-Agent": "opencode/1.0",
                "HTTP-Referer": "https://opencode.ai",
            },
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True},
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and line.startswith(b"data: "):
                data = line[6:]
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield _strip_think(delta)
                except (json.JSONDecodeError, IndexError):
                    continue

    def _stream_openrouter(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
        """Stream from OpenRouter."""
        import requests
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_get_key('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True},
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and line.startswith(b"data: "):
                data = line[6:]
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield _strip_think(delta)
                except (json.JSONDecodeError, IndexError):
                    continue

    def _stream_groq(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
        """Stream from Groq.

        Groq's streaming API for models like allam-2-7b and openai/gpt-oss-120b
        drops the leading whitespace from many token deltas. This means word-boundary
        spaces are lost when tokens are concatenated, producing concatenated words.

        Fix: request the FULL (non-streaming) response which always has correct spacing,
        then yield it in sentence-sized chunks to simulate streaming in the UI.
        """
        from groq import Groq
        client = Groq(api_key=_get_key("GROQ_API_KEY"))
        # Use non-streaming to guarantee correct whitespace
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=False,
        )
        full_text = _strip_think_full(resp.choices[0].message.content or "")
        if not full_text:
            return
        # Yield in ~60-char sentence chunks to simulate progressive display
        import re as _re
        chunks = _re.split(r'(?<=[.!?،؟])\s+', full_text)
        for ch in chunks:
            yield ch + " "

    def call(self, messages: List[Dict], role: str = "generate",
             model_override: Optional[str] = None, temperature: float = 0,
             max_tokens: int = 1200, response_format=None) -> Tuple[Optional[str], Optional[str]]:
        """
        Call LLM with automatic fallback. Returns (response_text, provider_name).
        
        Args:
            messages: Chat messages
            role: "scope", "critic", or "generate" (determines model)
            model_override: Force a specific model
            temperature: Sampling temperature
            max_tokens: Max tokens
            response_format: JSON response format (Groq only)
        """
        providers = self._get_available_providers()
        if not providers:
            return None, None

        for provider in providers:
            model = model_override or MODEL_DEFAULTS[provider].get(role, MODEL_DEFAULTS[provider]["generate"])
            try:
                if provider == "agentrouter":
                    text = self._call_agentrouter(messages, model, temperature, max_tokens)
                elif provider == "openrouter":
                    text = self._call_openrouter(messages, model, temperature, max_tokens)
                elif provider == "groq":
                    text = self._call_groq(messages, model, temperature, max_tokens, response_format)
                else:
                    continue

                text = _strip_think(text)
                self._last_provider = provider
                self._last_model = model
                logger.info(f"LLM call OK: provider={provider}, model={model}, role={role}")
                return text, provider

            except Exception as e:
                logger.warning(f"LLM call failed: provider={provider}, model={model}, error={e}")
                continue

        return None, None

    def stream(self, messages: List[Dict], role: str = "stream",
               model_override: Optional[str] = None, temperature: float = 0.1,
               max_tokens: int = 1200) -> Generator[Tuple[str, str], None, None]:
        """
        Stream LLM response token-by-token. Yields (token, provider_name).
        Falls back to non-streaming if streaming fails.
        """
        providers = self._get_available_providers()
        if not providers:
            yield ("⚠️ No LLM provider available.", "none")
            return

        for provider in providers:
            model = model_override or MODEL_DEFAULTS[provider].get(role, MODEL_DEFAULTS[provider]["stream"])
            try:
                if provider == "agentrouter":
                    gen = self._stream_agentrouter(messages, model, temperature, max_tokens)
                elif provider == "openrouter":
                    gen = self._stream_openrouter(messages, model, temperature, max_tokens)
                elif provider == "groq":
                    gen = self._stream_groq(messages, model, temperature, max_tokens)
                else:
                    continue

                self._last_provider = provider
                self._last_model = model
                logger.info(f"Streaming OK: provider={provider}, model={model}")
                for token in gen:
                    yield (token, provider)
                return

            except Exception as e:
                logger.warning(f"Stream failed: provider={provider}, model={model}, error={e}")
                # Fall back to non-streaming
                try:
                    text, src = self.call(messages, role="generate", model_override=model_override,
                                         temperature=temperature, max_tokens=max_tokens)
                    if text:
                        yield (text, src or provider)
                        return
                except Exception:
                    continue

        yield ("⚠️ All LLM providers failed.", "none")

    @property
    def last_provider(self) -> Optional[str]:
        return self._last_provider

    @property
    def last_model(self) -> Optional[str]:
        return self._last_model

    def status(self) -> Dict[str, bool]:
        """Return which providers are available."""
        return {
            "agentrouter": bool(_get_key("AGENTROUTER_API_KEY")),
            "openrouter": bool(_get_key("OPENROUTER_API_KEY")),
            "groq": bool(_get_key("GROQ_API_KEY")),
        }


# ─── Singleton for convenience ────────────────────────────────────────────────
_default_provider = None

def get_provider(preferred: Optional[str] = None) -> LLMProvider:
    """Get or create the default LLM provider singleton."""
    global _default_provider
    if _default_provider is None or preferred is not None:
        _default_provider = LLMProvider(preferred_provider=preferred)
    return _default_provider
