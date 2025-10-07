# Created by aaronkueh on 9/23/2025
# aom/tools/local_summarizer.py
from __future__ import annotations
import os
import re
import threading
from typing import Dict, Any
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from langchain_core.tools import tool
from aom.definitions import CONFIG_DIR

# -------- Lazy, shared loader (singleton) --------
_LOCK = threading.RLock()
_SLM: "LocalSummarizer | None" = None
_LLM_CFG: Dict[str, Any] | None = None

def _model_device(model) -> torch.device:
    # pick the device of the first parameter/buffer
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _trim_to_sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    m = list(re.finditer(r'([.!?])(?=\s|$)', s))
    return s if not m else s[:m[-1].end()].strip()

def _load_llm_yaml() -> Dict[str, Any]:
    global _LLM_CFG
    if _LLM_CFG is not None:
        return _LLM_CFG
    with _LOCK:
        if _LLM_CFG is not None:
            return _LLM_CFG
        path = os.path.join(CONFIG_DIR, "llm_config.yaml")
        try:
            with open(path, "r", encoding="utf-8") as f:
                _LLM_CFG = yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError):
            _LLM_CFG = {}
        return _LLM_CFG

def _llm_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Resolve profile: env override > meta.default_profile > single profile if only one > empty
    llm = (cfg or {}).get("llm", {})
    profiles = llm.get("profiles", {}) or {}
    if not profiles:
        return {}
    env_profile = os.getenv("LLM_PROFILE")
    profile_name = env_profile or (cfg.get("meta", {}) or {}).get("default_profile")
    if not profile_name and len(profiles) == 1:
        # auto-pick when only one exists
        profile_name = next(iter(profiles.keys()))
    return profiles.get(profile_name, {})

class NoDecimalCommaProcessor(LogitsProcessor):

    def __init__(self, tokenizer):
        self.tok = tokenizer
        # collect comma-like tokens your tokenizer may use
        self.block_ids = set()
        for s in [",", " ,", ", ", " , "]:
            ids = self.tok(s, add_special_tokens=False).input_ids
            if ids:
                self.block_ids.add(ids[0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # If the last visible char is a digit (optionally followed by a space), block comma tokens
        for i in range(scores.size(0)):
            txt = self.tok.decode(input_ids[i], skip_special_tokens=True)
            if re.search(r"\d\s?$", txt):
                for cid in self.block_ids:
                    scores[i, cid] = -float("inf")
        return scores

def no_decimal_comma(tokenizer):
    """Convenience helper: returns a LogitsProcessorList with the guard."""
    return LogitsProcessorList([NoDecimalCommaProcessor(tokenizer)])

class LocalSummarizer:
    """
    Small local LLM wrapper (default: SmolLM3-3B).
    Safe with device_map='auto'; does not force inputs to a fixed device.
    """
    def __init__(self, model_name: str = None, local_files_only: bool = False):
        cfg = _load_llm_yaml()
        prof = _llm_profile(cfg)

        # Resolve model_name: explicit arg > env > config > default
        cfg_model_name = (
            prof.get("model_name")
            or (cfg.get("meta", {}) or {}).get("model_name")
        )
        env_model = os.getenv("LLM_MODEL_NAME")
        self.model_name = model_name or env_model or cfg_model_name or "HuggingFaceTB/SmolLM3-3B"

        # Resolve local_files_only: HF_HUB_OFFLINE=1 wins; else config; else ctor arg
        offline_env = os.getenv("HF_HUB_OFFLINE")
        if offline_env is not None:
            self.local_files_only = offline_env == "1"
        else:
            self.local_files_only = bool(
                (cfg.get("meta", {}) or {}).get("local_files_only", local_files_only)
            )

        # Cache load/generation/tool defaults from config
        self.load_opts = {
            "device_map": (prof.get("load", {}) or {}).get("device_map", "auto"),
            "trust_remote_code": bool((prof.get("load", {}) or {}).get("trust_remote_code", False)),
            "revision": (prof.get("load", {}) or {}).get("revision"),
            "max_memory": (prof.get("load", {}) or {}).get("max_memory", None),
        }
        gen = (prof.get("generation", {}) or {})
        self.gen_defaults = {
            "max_new_tokens": int(gen.get("max_new_tokens", 640)),
            "min_new_tokens": int(gen.get("min_new_tokens", 32)),
            "temperature": float(gen.get("temperature", 0.2)),
            "top_p": float(gen.get("top_p", 0.9)),
            "do_sample": bool(gen.get("do_sample", False)),
            "no_repeat_ngram_size": int(gen.get("no_repeat_ngram_size", 3)),
            "repetition_penalty": float(gen.get("repetition_penalty", 1.05)),
        }
        tool_cfg = (prof.get("tool", {}) or {})
        self.tool_defaults = {
            "default_max_new_tokens": int(tool_cfg.get("default_max_new_tokens", self.gen_defaults["max_new_tokens"]))
        }

        self._tok = None
        self._model = None

    def _load_llm(self):
        if self._tok is not None and self._model is not None:
            return
        with _LOCK:
            if self._tok is not None and self._model is not None:
                return
            self._tok = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_files_only)
            model_kwargs = {
                "device_map": self.load_opts.get("device_map", "auto"),
                "local_files_only": self.local_files_only,
                "trust_remote_code": self.load_opts.get("trust_remote_code", False),
                "max_memory": self.load_opts.get("max_memory", None),
            }
            revision = self.load_opts.get("revision")
            if revision:
                model_kwargs["revision"] = revision

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            ).eval()
            # ensure pad token exists
            if self._tok.pad_token_id is None:
                self._tok.pad_token = self._tok.eos_token

    def summarize(
        self,
        prompt: str,
        max_new_tokens: int = 640,
        temperature: float = 0.2,
        top_p: float = 0.9,
        min_new_tokens: int = 32,
    ) -> str:
        self._load_llm()

        # Merge call-time args with config defaults (call-time wins when explicitly set)
        cfg_gen = self.gen_defaults
        max_new_tokens = max_new_tokens if max_new_tokens is not None else cfg_gen["max_new_tokens"]
        min_new_tokens = min_new_tokens if min_new_tokens is not None else cfg_gen["min_new_tokens"]
        temperature = temperature if temperature is not None else cfg_gen["temperature"]
        top_p = top_p if top_p is not None else cfg_gen["top_p"]

        inputs = self._tok(prompt, return_tensors="pt")
        # move inputs to the model's device (works with device_map="auto")
        dev = _model_device(self._model)
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        processors = no_decimal_comma(self._tok)

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=cfg_gen.get("do_sample", False),
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size=cfg_gen.get("no_repeat_ngram_size", 3),
                repetition_penalty=cfg_gen.get("repetition_penalty", 1.05),
                eos_token_id=self._tok.eos_token_id,
                pad_token_id=self._tok.pad_token_id,
                logits_processor=processors,
            )

        text = self._tok.decode(out_ids[0], skip_special_tokens=True)
        completion = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        return _trim_to_sentence(completion)

# -------- global accessor (so server/tools reuse the same instance) --------
def get_local_summarizer(local_files_only: bool = False) -> LocalSummarizer:
    global _SLM
    if _SLM is None:
        with _LOCK:
            if _SLM is None:
                # If caller passes a value, it overrides config/env; else use config/env
                _SLM = LocalSummarizer(local_files_only=local_files_only)
    return _SLM

# -------- LangChain tool wrapper (optional) --------
@tool("local_summarize")
def local_summarize(prompt: str, max_new_tokens: int = 640) -> str:
    """
    Summarize/analyze text using the local HF model (SmolLM3-3B by default).
    """
    slm = get_local_summarizer(local_files_only=os.getenv("HF_HUB_OFFLINE") == "1")
    # If the caller did not set max_new_tokens explicitly, use tool default from config
    max_tokens = max_new_tokens if max_new_tokens is not None else slm.tool_defaults["default_max_new_tokens"]
    return slm.summarize(prompt, max_new_tokens=max_tokens)
