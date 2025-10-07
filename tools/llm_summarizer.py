# Created by aaronkueh on 9/25/2025
# aom/tools/llm_summarizer.py
from __future__ import annotations
import os
import threading
from typing import Dict, Any
from langchain_core.tools import tool
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
import yaml
from pathlib import Path

_LOCK = threading.RLock()
_CLIENT: AsyncAnthropic | None = None
_LLM_CFG: Dict[str, Any] | None = None

CONFIG_DIR = (Path(__file__).resolve().parents[1] / "config")

def _load_llm_yaml() -> Dict[str, Any]:
    global _LLM_CFG
    if _LLM_CFG is not None:
        return _LLM_CFG
    with _LOCK:
        if _LLM_CFG is not None:
            return _LLM_CFG
        path = CONFIG_DIR / "llm_config.yaml"
        try:
            with path.open("r", encoding="utf-8") as f:
                _LLM_CFG = yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError):
            _LLM_CFG = {}
        return _LLM_CFG

def _llm_profile(cfg: Dict[str, Any], profile_name: str | None = None) -> Dict[str, Any]:
    meta = (cfg.get("meta", {}) or {})
    llm = (cfg.get("llm", {}) or {})
    profiles = (llm.get("profiles", {}) or {})
    name = profile_name or meta.get("default_profile") or "llm_claude"
    return profiles.get(name)

def _get_client() -> AsyncAnthropic:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _LOCK:
        if _CLIENT is not None:
            return _CLIENT
        load_dotenv()

        cfg = _load_llm_yaml()
        prof = _llm_profile(cfg)

        # Resolve env var names and API settings from config (with sensible defaults)
        env_cfg = (prof.get("env", {}) or {})
        api_cfg = (prof.get("api", {}) or {})

        api_key_env = env_cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} not set in environment or .env")

        base_url = api_cfg.get("base_url")
        version = api_cfg.get("version")  # typically '2023-06-01'

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if version:
            client_kwargs["default_headers"] = {"anthropic-version": str(version)}

        _CLIENT = AsyncAnthropic(**client_kwargs)
        return _CLIENT

class LLMSummarizer:
    def __init__(self, model: str | None = None):
        self.client = _get_client()

        cfg = _load_llm_yaml()
        prof = _llm_profile(cfg)

        # Determine model name: explicit arg > model_env var > config model_name > fallback
        cfg_model = (prof.get("model_name") or None)

        self.model = model or cfg_model

        # Cache generation defaults for convenience
        gen = (prof.get("generation", {}) or {})
        self.gen_defaults = {
            "max_new_tokens": int(gen.get("max_new_tokens", 640)),
            "temperature": float(gen.get("temperature", 0.2)),
            "top_p": float(gen.get("top_p", 0.7)),
        }

    async def summarize(self, prompt: str, max_new_tokens: int = 640) -> str:
        max_tokens = int(max_new_tokens if max_new_tokens is not None else self.gen_defaults["max_new_tokens"])
        msg = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.gen_defaults["temperature"],
            top_p=self.gen_defaults["top_p"],
        )
        print('msg content:\n', msg)
        # msg.content is a list of blocks; join text
        parts = []
        for b in getattr(msg, "content", []) or []:
            t = getattr(b, "text", None)
            if t:
                parts.append(t)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text") or "")
        text = "".join(parts).strip()
        return text

# -------- global accessor (so server/tools reuse the same instance) --------
_LLM: LLMSummarizer | None = None

def get_llm_summarizer() -> LLMSummarizer:
    global _LLM
    if _LLM is None:
        with _LOCK:
            if _LLM is None:
                _LLM = LLMSummarizer()
    return _LLM

# -------- LangChain tool wrapper --------
@tool("llm_summarize")
async def llm_summarize(prompt: str, max_new_tokens: int = 640) -> str:
    """
    Summarize/analyze text using Anthropic Claude.
    Reads model and API configuration from llm_config.yaml with env overrides.
    """
    cs = get_llm_summarizer()
    return await cs.summarize(prompt, max_new_tokens=max_new_tokens)