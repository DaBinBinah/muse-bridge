"""Emulated tool calling (TODO C3).

muse.ai has no native function calling and Muse, being an agent itself,
refuses prompts that ask it to "use" tools it does not have (verified against
the live web app). What it does cooperate with is a decision-only contract:
reply with a bare JSON object describing the calls. That is exactly the
protocol Muse-Chat-MCP proved against muse.ai, adopted here — OpenAI ``tools``
are described compactly in the prompt, a reply that is (optionally fenced)
``{"tool_calls": [{"name", "arguments"}]}`` is translated into standard
``tool_calls``, and later ``role: "tool"`` results are folded back into the
prompt as text turns labeled [Tool].

Hermes-style ``<tool_call>`` tag blocks are still accepted by the parser for
tolerance, but the prompt teaches the bare-JSON form. Everything here is
best-effort: a reply that does not match degrades to plain text.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

TOOL_TAG = "<tool_call>"
TOOL_END = "</tool_call>"

# A tag block ends at the closing tag; an unclosed block at end-of-text
# (truncated stream) still counts so a split call is not lost.
_TOOL_BLOCK_RE = re.compile(rf"{TOOL_TAG}\s*(.*?)\s*(?:{TOOL_END}|$)", re.DOTALL)

_INSTRUCTIONS = """If the request needs a tool, reply with ONLY this JSON and nothing else:
{{"tool_calls":[{{"name":"<tool name>","arguments":{{ ... }}}}]}}
You are only DECIDING whether a tool call should be made — do NOT perform the action, do NOT browse, do NOT invent results. The caller executes the tool and sends the result back to you as a message labeled "Tool"; use it to continue. Otherwise reply normally.

Tool schemas:
{spec}{choice}"""


TOOL_CONTINUATION = ('The Tool messages above are real results the caller obtained for you — '
                     'use them, do not re-check or re-run anything yourself. Reply now: either '
                     'the tool_calls JSON for one more call, or the final answer.')


def _tool_choice_line(tool_choice: Any) -> str:
    if tool_choice in (None, "auto"):
        return ""
    if tool_choice == "required":
        return '\nThis request MUST be answered with the tool_calls JSON: pick the most suitable tool now.'
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
        if name:
            return f'\nThis request MUST be answered with the tool_calls JSON calling the function "{name}".'
    return ""


def tool_instructions(tools: list[dict[str, Any]], tool_choice: Any = None) -> str:
    """System-style block describing the tools and the expected reply format."""
    spec = json.dumps(tools, ensure_ascii=False)
    return _INSTRUCTIONS.format(spec=spec, choice=_tool_choice_line(tool_choice))


def compose_tool_prompt(turns: list[tuple[str, str]], tools: list[dict[str, Any]],
                        tool_choice: Any = None) -> str:
    """Tool-request prompt for the live Muse agent, built from message turns.

    Roleplay transcripts with [User]/[Assistant]/[Tool] headers make the agent
    distrust the prompt ("that's a fake transcript"), so tool requests are
    narrated instead: the last user request, the real tool results that follow
    it, then the instructions — headerless, matching the single-message shape
    the agent already cooperates with.
    """
    instr = tool_instructions(tools, tool_choice)
    if not any(role == "tool" for role, _ in turns):
        text = "\n\n".join(t for _, t in turns)
        return f"{text}\n\n{instr}"
    last_user = next((t for role, t in reversed(turns) if role == "user"), "")
    results: list[str] = []
    decided = ""
    for role, t in reversed(turns):
        if role == "tool":
            results.append(t)
        else:
            # the assistant turn that produced the calls (rendered as JSON)
            if role == "assistant":
                decided = t
            break
    parts = [last_user]
    if decided:
        parts.append(f"You decided on:\n{decided}")
    parts.append("Tool result from the caller (real output of the call you decided on):\n"
                 + "\n\n".join(reversed(results)))
    parts.append(instr)
    return "\n\n".join(parts) + "\n" + TOOL_CONTINUATION


def render_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Assistant-history ``tool_calls`` back into the taught reply format, so
    the model recognises its own earlier calls."""
    calls = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        calls.append({"name": fn.get("name", ""), "arguments": args})
    return json.dumps({"tool_calls": calls}, ensure_ascii=False)


def _new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def _normalize_call(payload: dict[str, Any]) -> dict[str, Any] | None:
    """One parsed payload into an OpenAI ``tool_calls`` entry."""
    if not isinstance(payload, dict) or not payload.get("name"):
        return None
    args = payload.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {"id": _new_call_id(), "type": "function",
            "function": {"name": payload["name"], "arguments": args}}


def _strip_fences(payload: str) -> str:
    return re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", payload.strip())


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract tool calls from a model reply.

    Returns ``(content, calls)``: ``content`` is the reply with the calls
    removed (``None`` when the reply was only calls), ``calls`` the OpenAI
    ``tool_calls`` entries. Accepts the taught bare (optionally fenced)
    ``{"tool_calls": [...]}`` reply, Hermes-style ``<tool_call>`` tag blocks,
    parallel calls, a ``{"tool_calls": [...]}`` wrapper and truncated tag
    blocks; anything else passes through unchanged as plain text.
    """
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for m in _TOOL_BLOCK_RE.finditer(text):
        # Every detected block is stripped from the content — a half-written
        # call left in the reply helps nobody — but only parseable ones become
        # tool_calls; the rest degrade to a plain (shorter) text answer.
        spans.append(m.span())
        parsed = _loads_lenient(_strip_fences(m.group(1)))
        if isinstance(parsed, dict):
            call = _normalize_call(parsed)
            if call:
                calls.append(call)
        elif isinstance(parsed, list):
            calls.extend(c for c in (_normalize_call(p) for p in parsed
                                     if isinstance(p, dict)) if c)
    if not spans:
        # The taught format: the whole reply is one bare/fenced JSON object.
        parsed = _loads_lenient(_strip_fences(text))
        if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
            calls.extend(c for c in (_normalize_call(p) for p in parsed["tool_calls"]
                                     if isinstance(p, dict)) if c)
            if calls:
                return None, calls
        return text, []
    content = "".join(text[a:b] for a, b in zip([0] + [s for _, s in spans],
                                                [e for e, _ in spans] + [len(text)],
                                                strict=True))
    content = content.strip()
    return (content or None), calls


def _loads_lenient(payload: str) -> Any:
    try:
        parsed = json.loads(payload)
    except ValueError:
        return None
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except ValueError:
            return None
    return parsed


class ToolStreamFilter:
    """Stream gate for emulated tool-call replies.

    A tool reply is a bare (optionally fenced) JSON object spanning the whole
    reply, so the gate only has to sniff the first non-whitespace character:
    ``{`` or a ```` ``` ```` fence swallows the stream (the JSON becomes
    ``tool_calls`` deltas), anything else switches to pass-through. In
    pass-through a potential ``<tool_call>`` tag tail is still held back for
    tag-format models. ``finish()`` returns text that still belongs in the
    content: held-back plain text, or the swallowed payload when the JSON did
    not parse into calls.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._mode = "decide"  # decide -> passthrough | swallow

    @property
    def swallowing(self) -> bool:
        return self._mode == "swallow"

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        if self._mode == "swallow":
            self._buf += delta
            return ""
        self._buf += delta
        if self._mode == "decide":
            stripped = self._buf.lstrip()
            if not stripped:
                return ""
            if stripped[0] in "{`":
                self._mode = "swallow"
                return ""
            self._mode = "passthrough"
        # passthrough: hold back only a partial-tag tail
        idx = self._buf.find(TOOL_TAG)
        if idx != -1:
            out, self._buf = self._buf[:idx], self._buf[idx:]
            self._mode = "swallow"
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
        return out
