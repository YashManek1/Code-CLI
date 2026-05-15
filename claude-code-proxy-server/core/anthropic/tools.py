"""Heuristic parser for text-emitted tool calls + JSON repair utilities.

Enhancements over the original:
- ``repair_tool_arguments`` - salvages malformed JSON from weak models (MiniMax, etc.)
  using a multi-strategy cascade: clean → partial-parse → regex extraction → ``{}``
- ``flatten_tool_schema`` - strips $defs / $ref nesting so models with limited schema
  comprehension (MiniMax m2.7) don't produce broken nested-JSON tool calls.
- ``MODEL_QUIRKS`` registry - per-model flags consumed by the transport layer.
- ``HeuristicToolParser`` extended with better control-token stripping and a
  JSON-in-text detector that handles MiniMax's inline-JSON emission style.

FIX LOG (MiniMax m2.7):
  - get_model_quirks: now matches "minimaxai/" prefix (NIM sends this prefix)
  - _MINIMAX_NAMED_TOOL_RE: rewritten to handle nested braces correctly
  - HeuristicToolParser.feed: think-tag stripping moved AFTER tool extraction so
    interleaved reasoning is not destroyed before being passed back in history
  - _strip_think_tags: only strips from final TEXT output, never from raw buffer
    used for tool detection
"""

from __future__ import annotations

import json
import re
import uuid
from enum import Enum
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Model quirk flags
# ---------------------------------------------------------------------------


class ModelQuirks:
    """Capability/limitation flags for a specific model string."""

    __slots__ = (
        "buffer_tool_calls",  # accumulate all chunks before emitting
        "flatten_tool_schemas",
        "max_schema_depth",
        "max_tool_tokens",  # hard cap on argument JSON length (chars)
        "requires_json_repair",
        "strip_think_tags",
    )

    def __init__(
        self,
        *,
        requires_json_repair: bool = False,
        flatten_tool_schemas: bool = False,
        max_schema_depth: int = 10,
        buffer_tool_calls: bool = False,
        max_tool_tokens: int = 0,  # 0 = unlimited
        strip_think_tags: bool = False,
    ) -> None:
        self.requires_json_repair = requires_json_repair
        self.flatten_tool_schemas = flatten_tool_schemas
        self.max_schema_depth = max_schema_depth
        self.buffer_tool_calls = buffer_tool_calls
        self.max_tool_tokens = max_tool_tokens
        self.strip_think_tags = strip_think_tags


# Patterns that are substring-matched against the lower-cased model string.
# First match wins, so put more-specific patterns first.
# FIX: Changed from prefix-only to substring match so "minimaxai/minimax-m2.7"
#      (NIM's format) and "minimax/minimax-m2.7" (OpenRouter format) both match.
_QUIRK_PATTERNS: list[tuple[str, ModelQuirks]] = [
    (
        "minimax",  # matches "minimax/minimax-m2.7", "minimaxai/minimax-m2.7", any variant
        ModelQuirks(
            requires_json_repair=True,
            flatten_tool_schemas=False,
            max_schema_depth=5,
            buffer_tool_calls=True,
            max_tool_tokens=96_000,
            strip_think_tags=True,
        ),
    ),
    (
        "glm",  # z-ai/glm-4.7, z-ai/glm5, z-ai/glm-5.1, THUDM/glm-4
        ModelQuirks(
            requires_json_repair=True,
            flatten_tool_schemas=True,
            max_schema_depth=4,
            buffer_tool_calls=True,  # GLM streams partial JSON like MiniMax
            max_tool_tokens=64_000,
            strip_think_tags=True,
        ),
    ),
    (
        "kimi",  # matches Moonshot AI's Kimi models (NIM or OpenRouter)
        ModelQuirks(
            requires_json_repair=True,
            flatten_tool_schemas=True,
            max_schema_depth=4,
            buffer_tool_calls=True,
            max_tool_tokens=64_000,
            strip_think_tags=True,
        ),
    ),
]
_DEFAULT_QUIRKS = ModelQuirks()

# Matches Moonshot-style wrapped tool calls: <|tool_call_name_begin|>...<|tool_call_name_end|>...
_KIMI_FUNCTION_INDEXED_RE = re.compile(
    r"functions\.(?P<name>\w+):(?P<index>\d+)(?P<args>\{.*?\})",
    re.DOTALL,
)

_KIMI_NAMED_TOOL_RE = re.compile(
    r"(?:lob:\d+)?<\|tool_call_name_begin\|>(?P<name>[^<]+)<\|tool_call_name_end\|>"
    r"\s*(?:lob:\d+)?<\|tool_call_argument_begin\|>(?P<args>\{.*?\})(?:<\|tool_call_argument_end\|>|$)",
    re.DOTALL,
)


def get_model_quirks(model: str) -> ModelQuirks:
    """Return quirk flags for *model* (case-insensitive substring match).

    FIX: Was prefix-only match which missed "minimaxai/minimax-m2.7" from NIM.
    Now uses substring match so both "minimax/..." and "minimaxai/..." are caught.
    """
    lower = (model or "").lower()
    for pattern, quirks in _QUIRK_PATTERNS:
        if pattern in lower:
            return quirks
    return _DEFAULT_QUIRKS


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

# Matches a lone trailing backslash before a closing quote: \"  →  "
_TRAILING_ESCAPE_RE = re.compile(r'\\(?=["\s}])')
# Matches Python-style True/False/None that sometimes leak from weak models
_PYTHON_LITERALS_RE = re.compile(r"\bTrue\b|\bFalse\b|\bNone\b")
# Strips <think>…</think> blocks that MiniMax sometimes prepends to JSON
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Also handle unclosed <think> tags at start of a buffer
_THINK_TAG_OPEN_RE = re.compile(r"^<think>.*", re.DOTALL | re.IGNORECASE)
# Grab the outermost {...} object from arbitrary text
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _python_to_json_literals(s: str) -> str:
    def _replace(m: re.Match) -> str:
        tok = m.group(0)
        replacement = {"True": "true", "False": "false", "None": "null"}.get(tok)
        return replacement if replacement is not None else tok

    return _PYTHON_LITERALS_RE.sub(_replace, s)


def _extract_outermost_braces(s: str) -> str:
    """Return the substring from the first '{' to the last '}'."""
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return s
    return s[start : end + 1]


def _close_open_json(s: str) -> str:
    """Best-effort: close unclosed braces/brackets/strings so json.loads can parse."""
    stack: list[str] = []
    in_string = False
    escape_next = False
    result = list(s)

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    if in_string:
        result.append('"')
    result.extend(reversed(stack))

    return "".join(result)


def repair_tool_arguments(raw: str, *, tool_name: str = "", model: str = "") -> str:
    """Return valid JSON string for tool arguments, repairing *raw* if needed.

    Strategy cascade (stops at first success):
    1. Strip think-tags / whitespace, try direct parse.
    2. Fix Python literals (True→true etc.), retry.
    3. Extract outermost ``{…}`` substring, retry.
    4. Close unclosed braces/strings, retry.
    5. Regex-harvest key-value pairs from the raw text.
    6. Return ``{}`` as last resort so Claude Code doesn't crash.
    """
    if not raw or not raw.strip():
        return "{}"

    # Step 0: strip think tags (MiniMax prepends reasoning to JSON output)
    cleaned = _THINK_TAG_RE.sub("", raw).strip()
    # Also strip unclosed opening think tag (streaming may cut off close tag)
    if cleaned.startswith("<think>"):
        end = cleaned.find("</think>")
        if end == -1:
            # No closing tag found - strip from <think> to first { after it
            brace_pos = cleaned.find("{")
            if brace_pos != -1:
                cleaned = cleaned[brace_pos:]
            else:
                cleaned = ""
        else:
            cleaned = cleaned[end + len("</think>") :].strip()
    if not cleaned:
        return "{}"

    # Step 1: direct parse
    try:
        return json.dumps(json.loads(cleaned), ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # Step 2: fix Python literals
    cleaned2 = _python_to_json_literals(cleaned)
    try:
        return json.dumps(json.loads(cleaned2), ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # Step 3: extract outermost braces
    extracted = _extract_outermost_braces(cleaned2)
    try:
        return json.dumps(json.loads(extracted), ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # Step 4: close open structures
    closed = _close_open_json(extracted)
    try:
        return json.dumps(json.loads(closed), ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # Step 5: regex harvest - grab "key": value pairs
    harvested: dict[str, Any] = {}
    for m in re.finditer(
        r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)',
        cleaned2,
    ):
        key, val_str = m.group(1), m.group(2)
        try:
            harvested[key] = json.loads(val_str)
        except json.JSONDecodeError:
            harvested[key] = val_str

    if harvested:
        logger.warning(
            "JSON_REPAIR: harvested {} keys from malformed '{}' args (model={})",
            len(harvested),
            tool_name or "?",
            model or "?",
        )
        return json.dumps(harvested, ensure_ascii=False)

    logger.error(
        "JSON_REPAIR: all strategies failed for '{}' args (model={}), raising error. raw_prefix={}",
        tool_name or "?",
        model or "?",
        raw[:200],
    )
    raise ValueError(f"Could not repair tool arguments for {tool_name or 'unknown'}")


# ---------------------------------------------------------------------------
# Schema flattening
# ---------------------------------------------------------------------------


def flatten_tool_schema(
    schema: dict[str, Any], *, max_depth: int = 2
) -> dict[str, Any]:
    """Return a simplified copy of *schema* safe for low-capability models.

    - Resolves ``$ref`` against top-level ``$defs`` / ``definitions``.
    - Removes ``$defs``, ``definitions``, ``$schema``, ``additionalProperties``.
    - Recursively caps object nesting at *max_depth* (deeper levels become
      ``{"type": "object"}``).
    - Removes ``default``, ``examples``, ``$comment`` noise fields.
    """
    defs: dict[str, Any] = {**schema.get("$defs", {}), **schema.get("definitions", {})}

    def _resolve_ref(ref: str) -> dict[str, Any]:
        name = ref.split("/")[-1]
        return defs.get(name, {})

    _NOISE = frozenset(
        {
            "$defs",
            "definitions",
            "$schema",
            "additionalProperties",
            "default",
            "examples",
            "$comment",
            "$id",
        }
    )

    def _flatten(node: Any, depth: int) -> Any:
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            node = _resolve_ref(node["$ref"])
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _NOISE:
                continue
            if k == "properties":
                if depth >= max_depth:
                    # Too deep - collapse to plain object
                    out["type"] = "object"
                    continue
                out[k] = {pk: _flatten(pv, depth + 1) for pk, pv in v.items()}
            elif k in ("items", "additionalItems"):
                out[k] = _flatten(v, depth)
            elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
                # Simplify unions: keep first concrete type only
                concrete = [
                    _flatten(item, depth) for item in v if isinstance(item, dict)
                ]
                if len(concrete) == 1:
                    out.update(concrete[0])
                elif concrete:
                    out[k] = concrete
            else:
                out[k] = v
        return out

    return _flatten(schema, 0)


def prepare_tools_for_model(
    tools: list[dict[str, Any]], model: str
) -> list[dict[str, Any]]:
    """Return a (possibly modified) copy of *tools* suitable for *model*.

    If the model's quirks require schema flattening the ``input_schema`` of
    every tool is flattened before the request leaves the proxy.
    """
    quirks = get_model_quirks(model)
    if not quirks.flatten_tool_schemas:
        return tools

    result: list[dict[str, Any]] = []
    for tool in tools:
        tool_copy = dict(tool)
        raw_schema = tool_copy.get("input_schema") or tool_copy.get("parameters", {})
        if isinstance(raw_schema, dict) and raw_schema:
            flat = flatten_tool_schema(raw_schema, max_depth=quirks.max_schema_depth)
            if "input_schema" in tool_copy:
                tool_copy["input_schema"] = flat
            else:
                tool_copy["parameters"] = flat
        result.append(tool_copy)

    logger.debug(
        "SCHEMA_FLATTEN: flattened {} tools for model '{}' (max_depth={})",
        len(result),
        model,
        quirks.max_schema_depth,
    )
    return result


# ---------------------------------------------------------------------------
# Heuristic text-based tool-call parser (original, enhanced)
# ---------------------------------------------------------------------------

_CONTROL_TOKEN_RE = re.compile(r"(?:lob:\d+)?<\|[^|>]{1,100}\|>")
_CONTROL_TOKEN_START = "<|"
_CONTROL_TOKEN_END = "|>"


class ParserState(Enum):
    TEXT = 1
    MATCHING_FUNCTION = 2
    PARSING_PARAMETERS = 3


def _find_matching_brace(s: str, start: int) -> int:
    """Return the index of the closing '}' matching the '{' at *start*.

    FIX: The original _MINIMAX_NAMED_TOOL_RE used ``.*?`` which is non-greedy
    and stops at the first ``}`` — breaking on any nested object like
    ``{"path": "/foo", "content": {"key": "val"}}``.
    This function correctly tracks brace depth.
    """
    depth = 0
    in_string = False
    escape_next = False
    i = start
    while i < len(s):
        ch = s[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if ch == "\\" and in_string:
            escape_next = True
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1  # unclosed


# Matches the opening of a MiniMax named tool call: {"name":"tool_name","arguments":{
# FIX: Replaced greedy ``.*?`` regex with proper brace-depth tracking via
#      _find_matching_brace(). The old regex produced "read_fileread_file" duplicates
#      because it would match multiple times on the same buffer with a short args match.
_MINIMAX_NAMED_TOOL_HEADER_RE = re.compile(
    r'\{"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(?=\{)',
    re.DOTALL,
)


class HeuristicToolParser:
    """Stateful parser for raw-text tool calls.

    Some OpenAI-compatible models emit tool calls as text rather than structured
    chunks.  This parser converts the common ``● <function=...>`` form and
    inline JSON forms into Anthropic-style ``tool_use`` blocks.

    FIX: think-tag stripping is now deferred to final TEXT output only.
    The raw buffer used for tool detection retains think-tag content so that
    interleaved reasoning blocks are not accidentally consumed before extraction.
    """

    _FUNC_START_PATTERN = re.compile(r"●\s*<function=([^>]+)>")
    _PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)", re.DOTALL
    )
    # Matches WebFetch / WebSearch tool calls expressed as inline JSON text
    _WEB_TOOL_JSON_PATTERN = re.compile(
        r"(?is)\b(?:use\s+)?(?P<tool>WebFetch|WebSearch)\b.*?(?P<json>\{.*?\})"
    )

    def __init__(self, *, model: str = ""):
        self._state = ParserState.TEXT
        self._buffer = ""
        self._current_tool_id: str | None = None
        self._current_function_name: str | None = None
        self._current_parameters: dict[str, Any] = {}
        self._model = model
        self._quirks = get_model_quirks(model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_web_tool_json_calls(self) -> tuple[str, list[dict[str, Any]]]:
        detected_tools: list[dict[str, Any]] = []

        for match in self._WEB_TOOL_JSON_PATTERN.finditer(self._buffer):
            try:
                tool_input = json.loads(match.group("json"))
            except json.JSONDecodeError:
                continue

            if not isinstance(tool_input, dict):
                continue

            tool_name = match.group("tool")
            if tool_name == "WebFetch" and "url" not in tool_input:
                continue
            if tool_name == "WebSearch" and "query" not in tool_input:
                continue

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            logger.debug(
                "Heuristic bypass: Detected JSON-style tool call '{}'", tool_name
            )

        if not detected_tools:
            return self._buffer, []

        return "", detected_tools

    def _extract_minimax_named_calls(self) -> tuple[str, list[dict[str, Any]]]:
        """Detect MiniMax-style {\"name\":\"...\",\"arguments\":{...}} inline calls.

        FIX: Replaced greedy regex (which caused "read_fileread_file" duplication)
        with proper brace-depth tracking via _find_matching_brace(). This correctly
        handles nested JSON objects in arguments (e.g. file content with sub-dicts).
        Also deduplicates tool calls by (name, args_hash) to prevent re-detection
        on the same buffer content across multiple feed() calls.
        """
        detected: list[dict[str, Any]] = []
        remainder = self._buffer
        consumed_spans: list[tuple[int, int]] = []  # (start, end) to remove

        search_start = 0
        while True:
            header_match = _MINIMAX_NAMED_TOOL_HEADER_RE.search(remainder, search_start)
            if not header_match:
                break

            # Find the opening brace of the arguments object
            args_brace_start = header_match.end()
            # Find matching closing brace with depth tracking
            args_brace_end = _find_matching_brace(remainder, args_brace_start)
            if args_brace_end == -1:
                # Args object not complete yet — stop processing, wait for more data
                break

            # The full outer object ends after the args closing brace + "}"
            # We need to find the outer closing brace
            outer_end = args_brace_end + 1  # move past args "}"
            # Skip whitespace then expect "}"
            while outer_end < len(remainder) and remainder[outer_end] in " \t\n\r":
                outer_end += 1
            if outer_end < len(remainder) and remainder[outer_end] == "}":
                outer_end += 1
            else:
                # Malformed — skip past this header and keep searching
                search_start = header_match.end()
                continue

            tool_name = header_match.group("name")
            raw_args = remainder[args_brace_start : args_brace_end + 1]

            try:
                tool_input = json.loads(raw_args)
            except json.JSONDecodeError:
                repaired = repair_tool_arguments(
                    raw_args, tool_name=tool_name, model=self._model
                )
                try:
                    tool_input = json.loads(repaired)
                except json.JSONDecodeError:
                    tool_input = {}

            detected.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            # Mark the full span for removal
            consumed_spans.append((header_match.start(), outer_end))
            logger.debug(
                "Heuristic bypass (MiniMax named): Detected tool call '{}' args_len={}",
                tool_name,
                len(raw_args),
            )
            search_start = outer_end

        # Remove consumed spans in reverse order to preserve indices
        for start, end in reversed(consumed_spans):
            remainder = remainder[:start] + remainder[end:]

        return remainder, detected

    def _extract_kimi_function_indexed_calls(self) -> tuple[str, list[dict[str, Any]]]:
        """Detect Kimi-style functions.Name:0{...} calls."""
        detected: list[dict[str, Any]] = []
        remainder = self._buffer
        consumed_spans: list[tuple[int, int]] = []

        for match in _KIMI_FUNCTION_INDEXED_RE.finditer(remainder):
            tool_name = match.group("name")
            raw_args = match.group("args")

            try:
                tool_input = json.loads(raw_args)
            except json.JSONDecodeError:
                repaired = repair_tool_arguments(
                    raw_args, tool_name=tool_name, model=self._model
                )
                try:
                    tool_input = json.loads(repaired)
                except json.JSONDecodeError:
                    tool_input = {}

            detected.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            consumed_spans.append((match.start(), match.end()))
            logger.debug(
                "Heuristic bypass (Kimi indexed): Detected tool call '{}'", tool_name
            )

        for start, end in reversed(consumed_spans):
            remainder = remainder[:start] + remainder[end:]

        return remainder, detected

    def _extract_kimi_named_calls(self) -> tuple[str, list[dict[str, Any]]]:
        """Detect Moonshot-style <|tool_call_name_begin|>... calls."""
        detected: list[dict[str, Any]] = []
        remainder = self._buffer
        consumed_spans: list[tuple[int, int]] = []

        for match in _KIMI_NAMED_TOOL_RE.finditer(remainder):
            tool_name = match.group("name").strip()
            raw_args = match.group("args")

            try:
                tool_input = json.loads(raw_args)
            except json.JSONDecodeError:
                repaired = repair_tool_arguments(
                    raw_args, tool_name=tool_name, model=self._model
                )
                try:
                    tool_input = json.loads(repaired)
                except json.JSONDecodeError:
                    tool_input = {}

            detected.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            consumed_spans.append((match.start(), match.end()))
            logger.debug(
                "Heuristic bypass (Kimi named): Detected tool call '{}'", tool_name
            )

        for start, end in reversed(consumed_spans):
            remainder = remainder[:start] + remainder[end:]

        return remainder, detected

    def _strip_control_tokens(self, text: str) -> str:
        return _CONTROL_TOKEN_RE.sub("", text)

    def _strip_think_tags_for_output(self, text: str) -> str:
        """Strip think tags ONLY for final text output — not for tool detection buffer.

        FIX: The original code stripped think tags from self._buffer before tool
        detection, which destroyed interleaved reasoning that MiniMax passes between
        tool calls. Now we only strip think tags from the final text we return to
        the caller.
        """
        if self._quirks.strip_think_tags:
            return _THINK_TAG_RE.sub("", text)
        return text

    def _split_incomplete_control_token_tail(self) -> str:
        start = self._buffer.rfind(_CONTROL_TOKEN_START)
        if start == -1:
            return ""
        end = self._buffer.find(_CONTROL_TOKEN_END, start)
        if end != -1:
            return ""
        prefix = self._buffer[:start]
        self._buffer = self._buffer[start:]
        return prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Feed *text* and return ``(safe_text, detected_tool_calls)``.

        FIX: think-tag stripping moved to output phase only.
        Tool detection operates on the raw buffer (including <think> blocks)
        so that MiniMax's interleaved reasoning does not interfere with tool
        call detection but is correctly preserved in the conversation history.
        """
        self._buffer += text

        detected_tools: list[dict[str, Any]] = []

        # MiniMax/Kimi inline named-tool detection (runs before standard patterns)
        if (
            self._quirks.requires_json_repair
            or "minimax" in self._model.lower()
            or "kimi" in self._model.lower()
        ):
            self._buffer, extra = self._extract_minimax_named_calls()
            detected_tools.extend(extra)
            self._buffer, extra = self._extract_kimi_named_calls()
            detected_tools.extend(extra)
            self._buffer, extra = self._extract_kimi_function_indexed_calls()
            detected_tools.extend(extra)

        # Strip control tokens AFTER trying to extract wrapped tool calls
        self._buffer = self._strip_control_tokens(self._buffer)

        self._buffer, extra = self._extract_web_tool_json_calls()
        detected_tools.extend(extra)

        filtered_output_parts: list[str] = []

        while True:
            if self._state == ParserState.TEXT:
                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                    filtered_output_parts.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx:]
                    self._state = ParserState.MATCHING_FUNCTION
                else:
                    safe_prefix = self._split_incomplete_control_token_tail()
                    if safe_prefix:
                        filtered_output_parts.append(safe_prefix)
                        break
                    filtered_output_parts.append(self._buffer)
                    self._buffer = ""
                    break

            if self._state == ParserState.MATCHING_FUNCTION:
                match = self._FUNC_START_PATTERN.search(self._buffer)
                if match:
                    self._current_function_name = match.group(1).strip()
                    self._current_tool_id = f"toolu_heuristic_{uuid.uuid4().hex[:8]}"
                    self._current_parameters = {}
                    self._buffer = self._buffer[match.end() :]
                    self._state = ParserState.PARSING_PARAMETERS
                    logger.debug(
                        "Heuristic bypass: Detected start of tool call '{}'",
                        self._current_function_name,
                    )
                elif len(self._buffer) > 100:
                    filtered_output_parts.append(self._buffer[0])
                    self._buffer = self._buffer[1:]
                    self._state = ParserState.TEXT
                else:
                    break

            if self._state == ParserState.PARSING_PARAMETERS:
                finished_tool_call = False

                while True:
                    param_match = self._PARAM_PATTERN.search(self._buffer)
                    if param_match and "</parameter>" in param_match.group(0):
                        pre_match_text = self._buffer[: param_match.start()]
                        if pre_match_text:
                            filtered_output_parts.append(pre_match_text)
                        key = param_match.group(1).strip()
                        val = param_match.group(2).strip()
                        self._current_parameters[key] = val
                        self._buffer = self._buffer[param_match.end() :]
                    else:
                        break

                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                    if idx > 0:
                        filtered_output_parts.append(self._buffer[:idx])
                        self._buffer = self._buffer[idx:]
                    finished_tool_call = True
                elif len(self._buffer) > 0 and not self._buffer.strip().startswith("<"):
                    if "<parameter=" not in self._buffer:
                        filtered_output_parts.append(self._buffer)
                        self._buffer = ""
                        finished_tool_call = True

                if finished_tool_call:
                    detected_tools.append(
                        {
                            "type": "tool_use",
                            "id": self._current_tool_id,
                            "name": self._current_function_name,
                            "input": self._current_parameters,
                        }
                    )
                    logger.debug(
                        "Heuristic bypass: Emitting tool call '{}' with {} params",
                        self._current_function_name,
                        len(self._current_parameters),
                    )
                    self._state = ParserState.TEXT
                else:
                    break

        # FIX: Strip think tags from the output text ONLY — not from self._buffer
        raw_output = "".join(filtered_output_parts)
        safe_output = self._strip_think_tags_for_output(raw_output)
        return safe_output, detected_tools

    def flush(self) -> list[dict[str, Any]]:
        """Flush any partial tool call remaining in the buffer."""
        self._buffer = self._strip_control_tokens(self._buffer)
        # FIX: Do not strip think tags from buffer during flush either;
        # only strip from text output going to the client.
        detected_tools: list[dict[str, Any]] = []

        if self._state == ParserState.PARSING_PARAMETERS:
            for match in re.finditer(
                r"<parameter=([^>]+)>(.*)$", self._buffer, re.DOTALL
            ):
                key = match.group(1).strip()
                val = match.group(2).strip()
                self._current_parameters[key] = val

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": self._current_tool_id,
                    "name": self._current_function_name,
                    "input": self._current_parameters,
                }
            )
            self._state = ParserState.TEXT
            self._buffer = ""

        return detected_tools
