# Created by aaronkueh on 9/23/2025
# aom/utils/agent_helpers.py

from typing import Dict, Any, Tuple, Optional
# from aom.mcp_server.server import call_tool
# from transformers import LogitsProcessor
# import torch
import re, unicodedata


def slm_summarize(
    prompt: str,
    *,
    max_new_tokens: int = 640,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """
    Call the MCP-registered local summarizer tool and return plain text.
    - Expects server to have registered 'local_summarize'.
    - Returns an empty string on missing output (doesn't raise).
    """
    from aom.mcp_server.server import call_tool

    payload: Dict[str, Any] = {"prompt": prompt, "max_new_tokens": max_new_tokens}

    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    res = call_tool("local_summarize", payload)
    # server-side wrapper returns {"text": "..."}
    return (res or {}).get("text", "").strip()

def trim_sentence(text: str) -> str:
    """
    Return text cut at the last completed sentence.
    A 'completed sentence' ends with ., !, or ? and may be followed by closing quotes/brackets.
    If no terminal punctuation is found, returns the original string stripped.
    """
    s = (text or "").strip()
    if not s:
        return s
    # Find last sentence-ending punctuation; allow closing quotes/brackets after it.
    matches = list(re.finditer(r'([.!?])(?:["\')\]]+)?', s))
    if not matches:
        return s
    return s[: matches[-1].end()].strip()


def count_tokens(text: str, model_name: Optional[str] = None) -> Tuple[int, bool]:
    """
    Count tokens for the given text with the model's tokenizer if available.
    Returns (token_count, exact) where:
      - token_count: number of tokens
      - exact: True if computed using the tokenizer; False if estimated

    Fallback (no tokenizer available): estimate 1 token ≈ 3.5 characters.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name or "HuggingFaceTB/SmolLM3-3B")
        return int(tok(text, return_tensors="pt")["input_ids"].shape[1]), True
    except ImportError:
        return int(len(text) / 3.5), False


def token_budget_info(
    total_window_hint: int = 8192,
    max_new_tokens: int = 640,
    safety_buffer: int = 50,
) -> Tuple[int, int, int]:
    """
    Compute a safe input budget from a total window hint.
    Returns (safe_input_budget, total_window_hint, max_new_tokens).
    """
    safe_input = max(0, total_window_hint - max_new_tokens - safety_buffer)
    return safe_input, total_window_hint, max_new_tokens

def sanitize_llm_text(
    text: str,
    *,
    ascii_only: bool = False,
    strip_markdown: bool = True,
    remove_backslashes: bool = True,
) -> str:
    """
    Clean up LLM output for plain-text delivery.
    - Normalizes Unicode (NFKC) and whitespace.
    - Optionally strips simple Markdown markers (** __ ` ``` #).
    - Standardizes punctuation (smart quotes → ASCII quotes, dashes → '-').
    - Removes zero-width/control characters.
    - Optionally removes backslashes '\' which small models sometimes emit as escape noise.
    - Preserves the degree symbol by default; if ascii_only=True, replaces '°' with ' deg '.
    - Removes bracket characters while keeping their content.
    - Fixes spaced decimals like '4 . 5' → '4.5'.
    """
    if not text:
        return ""

    # Normalize Unicode form
    s = unicodedata.normalize("NFKC", text)

    # Remove zero-width and control characters (except basic whitespace)
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)              # zero-widths
    s = "".join(ch for ch in s if (ch == "\n" or ch == "\t" or ch >= " "))

    if strip_markdown:
        # Remove code fences/backticks, bold/italic markers, headings
        s = re.sub(r"`{3,}.*?\n", "", s, flags=re.DOTALL)     # ``` blocks
        s = re.sub(r"`+", "", s)                              # inline code ticks
        s = re.sub(r"(\*\*|__)", "", s)                       # bold markers
        # Leading list/star bullets commonly emitted by small models
        s = re.sub(r"^[\s>*\-•]+\s*", "", s, flags=re.MULTILINE)
        # Remove Markdown headings (e.g., "## Title")
        s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.MULTILINE)

    # Replace smart quotes and dashes with plain versions
    tbl = {
        "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
        "\u201C": '"', "\u201D": '"', "\u201E": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-",  # en/em/minus
        "\u2026": "...",  # ellipsis
    }
    s = s.translate(str.maketrans(tbl))

    # Optionally remove backslashes (escape noise)
    if remove_backslashes:
        s = s.replace("\\", "")

    # Remove all asterisks
    s = s.replace("*", "")
    # Remove underscores
    s = s.replace("_", "")

    # Remove bracket characters but keep their contents
    # ( ... ), [ ... ], { ... }, < ... >
    s = re.sub(r"[()\[\]{}<>]", "", s)

    # Fix spaced decimals: "4 . 5" or "4 .5" or "4. 5" -> "4.5"
    s = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", s)

    # Normalize spaces around punctuation (but respect decimals already fixed)
    s = re.sub(r"\s+([,;:!?])", r"\1", s)          # no space before common punctuation
    s = re.sub(r"([,;:!?])(?!\s|\Z)", r"\1 ", s)   # ensure the following space (except end)
    s = re.sub(r"\s{2,}", " ", s)                  # collapse multiple spaces
    # Keep single blank lines; collapse longer runs
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    # Degree symbol handling
    if ascii_only:
        s = s.replace("°", " deg ")
        # After replacing degree, collapse spaces again
        s = re.sub(r"\s{2,}", " ", s).strip()
        # Force ASCII by dropping non-ASCII remnants
        s = s.encode("ascii", "ignore").decode("ascii")

    return s
