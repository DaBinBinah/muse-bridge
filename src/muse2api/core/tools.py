"""Emulated tool calling (TODO C3).

muse.ai has no native function calling, so OpenAI ``tools`` are described in the
prompt using the Hermes ``<tool_call>`` convention (the format Qwen models are
trained on and vLLM parses natively). The model's reply is parsed back into
OpenAI ``tool_calls`` and later ``role: "tool"`` results are folded into the
prompt as text turns.

Everything here is best-effort: a reply that does not contain the marker is
returned unchanged as plain text, so the worst case is a normal chat answer.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

TOOL_TAG = "<tool_call>"
TOOL_END = "</tool_call>"

# A block ends at the closing tag; an unclosed block at end-of-text (truncated
# stream) still counts so a split call is not lost.
_TOOL_BLOCK_RE = re.compile(rf"{TOOL_TAG}\s*(.*?)\s*(?:{TOOL_END}|$)", re.DOTALL)

_INSTRUCTIONS = """# Tool use

The calling client can execute external tools for you. Tool definitions:

{spec}

When you decide a tool should be called, reply with ONLY tool call blocks, one
per call, in exactly this format:

{tag}
{{"name": "tool_name", "arguments": {{...}}}}
{end}

Rules:
- Decide only. Never claim a tool was run and never invent its result. The
  client executes the call and returns the result as a message labeled "Tool";
  continue from there.
- Call a tool only when it helps answer the request. If no tool is needed,
  answer normally in plain text without any tool call block.
- "arguments" must be a valid JSON object matching the tool's parameters
  schema; omit unknown fields.
{choice}"""


def _tool_choice_line(tool_choice: Any) -> str:
    if tool_choice in (None, "auto"):
        return ""
    if tool_choice == "required":
        return '- You MUST call at least one tool now; reply only with tool call blocks.'
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
        if name:
            return f'- You MUST call the function "{name}" now; reply only with a tool call block.'
    return ""


def tool_instructions(tools: list[dict[str, Any]], tool_choice: Any = None) -> str:
    """System-style block describing the tools and the expected reply format."""
    spec = json.dumps(tools, ensure_ascii=False, indent=2)
    return _INSTRUCTIONS.format(spec=spec, tag=TOOL_TAG, end=TOOL_END,
                                choice=_tool_choice_line(tool_choice))


def render_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Assistant-history ``tool_calls`` back into the marker format, so the model
    sees its own earlier calls exactly the way it was taught to write them."""
    blocks = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        blocks.append(TOOL_TAG + "\n" + json.dumps(
            {"name": fn.get("name", ""), "arguments": args}, ensure_ascii=False) + "\n" + TOOL_END)
    return "\n".join(blocks)


def _new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def _normalize_call(payload: dict[str, Any]) -> dict[str, Any] | None:
    """One parsed payload (str JSON or dict) into an OpenAI ``tool_calls`` entry."""
    if not isinstance(payload, dict) or not payload.get("name"):
        return None
    args = payload.get("arguments", {})
    if isinstance(args, dict):
        args = json.dumps(args, ensure_ascii=False)
    elif not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {"id": _new_call_id(), "type": "function",
            "function": {"name": payload["name"], "arguments": args}}


def _strip_fences(payload: str) -> str:
    return re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", payload.strip())


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract tool calls from a model reply.

    Returns ``(content, calls)``: ``content`` is the reply with all call blocks
    removed (``None`` when nothing but calls), ``calls`` the OpenAI
    ``tool_calls`` entries. Tolerates ```json fences inside the block, several
    blocks (parallel calls), a ``{"tool_calls": [...]}`` wrapper and a
    truncated final block.
    """
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for m in _TOOL_BLOCK_RE.finditer(text):
        # Every detected block is stripped from the content — a half-written
        # call left in the reply helps nobody — but only parseable ones become
        # tool_calls; the rest degrade to a plain (shorter) text answer.
        spans.append(m.span())
        payload = _strip_fences(m.group(1))
        try:
            parsed = json.loads(payload)
        except ValueError:
            parsed = None
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except ValueError:
                parsed = None
        if isinstance(parsed, dict) and not parsed.get("name") \
                and isinstance(parsed.get("tool_calls"), list):
            parsed = parsed["tool_calls"]
        if isinstance(parsed, dict):
            call = _normalize_call(parsed)
            if call:
                calls.append(call)
        elif isinstance(parsed, list):
            calls.extend(c for c in (_normalize_call(p) for p in parsed
                                     if isinstance(p, dict)) if c)
    if not spans:
        return text, []
    content = "".join(text[a:b] for a, b in zip([0] + [s for _, s in spans],
                                                [e for e, _ in spans] + [len(text)],
                                                strict=True))
    content = content.strip()
    return (content or None), calls


class ToolStreamFilter:
    """Stream gate for the marker tag.

    Passes text through while it cannot be the start of ``<tool_call>``,
    holding back only a potential partial tag at the chunk tail; once the tag
    appears the rest of the stream is swallowed (the call is converted to
    ``tool_calls`` deltas by the caller instead of being streamed as text).
    """

    def __init__(self) -> None:
        self._buf = ""
        self._tagged = False

    @property
    def tagged(self) -> bool:
        return self._tagged

    def feed(self, delta: str) -> str:
        if self._tagged or not delta:
            return "" if self._tagged else delta
        self._buf += delta
        idx = self._buf.find(TOOL_TAG)
        if idx != -1:
            self._tagged = True
            out = self._buf[:idx]
            self._buf = ""
            return out
        keep = 0
        for k in range(min(len(self._buf), len(TOOL_TAG) - 1), 0, -1):
            if TOOL_TAG.startswith(self._buf[-k:]):
                keep = k
                break
        if keep:
            out, self._buf = self._buf[:-keep], self._buf[-keep:]
        else:
            out, self._buf = self._buf, ""
        return out

    def finish(self) -> str:
        out, self._buf = self._buf, ""
        return out if not self._tagged else ""
