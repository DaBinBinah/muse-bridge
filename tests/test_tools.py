"""Emulated tool calling (TODO C3): prompt injection, parsing, stream gate and
the two OpenAI endpoints translating calls back and forth.

The taught protocol is the bare-JSON decision-only reply proven against the
live muse.ai agent ("reply with ONLY this JSON ... only DECIDING"); Hermes
``<tool_call>`` tag blocks remain accepted by the parser for tolerance.
"""

from __future__ import annotations

import json

import pytest

from muse2api.core.prompt import flatten_messages, message_turns
from muse2api.core.tools import TOOL_TAG, ToolStreamFilter, parse_tool_calls, tool_instructions

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

CANNED = '{"tool_calls":[{"name":"get_weather","arguments":{"city":"Beijing"}}]}'


# ---------- unit: parse_tool_calls (taught bare-JSON format) ----------

def test_parse_bare_json_reply():
    content, calls = parse_tool_calls(CANNED)
    assert content is None
    assert len(calls) == 1
    call = calls[0]
    assert call["id"].startswith("call_") and call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Beijing"}


def test_parse_fenced_bare_json_reply():
    content, calls = parse_tool_calls("```json\n" + CANNED + "\n```")
    assert content is None and len(calls) == 1


def test_parse_bare_json_without_tool_calls_key_passes_through():
    text = '{"answer": "just data, not a tool call"}'
    content, calls = parse_tool_calls(text)
    assert content == text and calls == []


def test_parse_bare_parallel_calls():
    content, calls = parse_tool_calls(
        '{"tool_calls":[{"name":"a","arguments":{}},{"name":"b","arguments":{"x":1}}]}')
    assert content is None
    assert [c["function"]["name"] for c in calls] == ["a", "b"]


# ---------- unit: parse_tool_calls (tag tolerance) ----------

def test_parse_tag_block_with_preamble():
    text = f"Let me check.\n{TOOL_TAG}\n" \
           '{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>'
    content, calls = parse_tool_calls(text)
    assert content == "Let me check."
    assert len(calls) == 1


def test_parse_tag_block_gives_none_content():
    content, calls = parse_tool_calls(
        f'{TOOL_TAG}\n{{"name": "f", "arguments": {{}}}}\n</tool_call>')
    assert content is None and len(calls) == 1


def test_parse_tag_parallel_calls():
    text = (f'{TOOL_TAG}\n{{"name": "a", "arguments": {{}}}}\n</tool_call>\n'
            f'{TOOL_TAG}\n{{"name": "b", "arguments": {{"x": 1}}}}\n</tool_call>')
    content, calls = parse_tool_calls(text)
    assert content is None
    assert [c["function"]["name"] for c in calls] == ["a", "b"]


def test_parse_tag_fenced_json_inside_block():
    text = f'{TOOL_TAG}\n```json\n{{"name": "a", "arguments": {{}}}}\n```\n</tool_call>'
    _, calls = parse_tool_calls(text)
    assert calls and calls[0]["function"]["name"] == "a"


def test_parse_truncated_tag_block_stripped():
    text = f'Checking.{TOOL_TAG}\n{{"name": "a", "arguments": {{"x"'
    content, calls = parse_tool_calls(text)
    assert content == "Checking." and calls == []


def test_parse_truncated_tag_block_with_valid_json():
    text = f'{TOOL_TAG}\n{{"name": "a", "arguments": {{}}}}'
    content, calls = parse_tool_calls(text)
    assert content is None and len(calls) == 1


def test_parse_plain_text_untouched():
    text = "No tools needed, here is the answer."
    content, calls = parse_tool_calls(text)
    assert content == text and calls == []


def test_parse_string_arguments_kept_as_string():
    text = f'{TOOL_TAG}\n{{"name": "a", "arguments": "{{\\"k\\": 1}}"}}\n</tool_call>'
    _, calls = parse_tool_calls(text)
    assert calls[0]["function"]["arguments"] == '{"k": 1}'


# ---------- unit: ToolStreamFilter ----------

def test_stream_filter_swallows_bare_json():
    f = ToolStreamFilter()
    assert f.feed('{"tool') == ""
    assert f.feed('_calls": ...') == ""
    assert f.feed("}") == ""
    assert f.swallowing
    assert f.finish() == '{"tool_calls": ...}'


def test_stream_filter_passes_clean_text():
    f = ToolStreamFilter()
    out = f.feed("hello world") + f.feed("!")
    assert out == "hello world!"
    assert f.finish() == ""


def test_stream_filter_leading_whitespace_held_then_plain():
    f = ToolStreamFilter()
    assert f.feed("  \n") == ""
    out = f.feed("plain text")
    assert out == "  \nplain text"


def test_stream_filter_split_tag():
    f = ToolStreamFilter()
    assert f.feed("Answer: 42. <tool") == "Answer: 42. "
    assert f.feed("_call>\nsecret") == ""
    assert f.swallowing
    # the swallowed payload is returned by finish(); the route drops it when
    # the final parse produced calls
    assert "secret" in f.finish()


def test_stream_filter_holds_partial_tag_tail():
    f = ToolStreamFilter()
    assert f.feed("tail <tool_c") == "tail "
    assert f.finish() == "<tool_c"


# ---------- unit: tool_instructions ----------

def test_instructions_embed_schema_and_rules():
    text = tool_instructions(TOOLS)
    assert '"get_weather"' in text
    assert "only DECIDING" in text
    assert "do NOT perform the action" in text
    assert '"tool_calls"' in text


@pytest.mark.parametrize("choice, needle", [
    ("required", "MUST be answered with the tool_calls JSON"),
    ({"type": "function", "function": {"name": "get_weather"}},
     'tool_calls JSON calling the function "get_weather"'),
    ("auto", None),
])
def test_instructions_tool_choice(choice, needle):
    text = tool_instructions(TOOLS, choice)
    if needle:
        assert needle in text
    else:
        assert "MUST" not in text


# ---------- unit: prompt flattening ----------

def test_flatten_renders_tool_history():
    messages = [
        {"role": "user", "content": "weather in Beijing?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather",
                          "arguments": "{\"city\": \"Beijing\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 24C"},
    ]
    flat = flatten_messages(messages)
    assert "[Tool]" in flat.text
    assert '"get_weather"' in flat.text
    assert "Sunny, 24C" in flat.text
    turns = message_turns(messages)
    assert ("tool", "Sunny, 24C") in turns


# ---------- integration helpers ----------

def _stub_chat(client, monkeypatch, reply: str):
    """Canned upstream reply, chunked like a real stream; returns captured requests."""
    gateway = client.app.state.services.gateway
    seen = []

    async def fake(req):
        seen.append(req)
        for i in range(0, len(reply), 7):
            yield reply[i:i + 7]

    monkeypatch.setattr(gateway, "chat_stream", fake)
    return seen


# ---------- integration: /v1/chat/completions ----------

def test_chat_tools_non_stream(client, auth, monkeypatch):
    _stub_chat(client, monkeypatch, CANNED)
    r = client.post("/v1/chat/completions", headers=auth, json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "weather?"}],
        "tools": TOOLS})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    msg = choice["message"]
    assert msg["content"] is None
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Beijing"}


def test_chat_tools_prompt_injection(client, auth, monkeypatch):
    seen = _stub_chat(client, monkeypatch, CANNED)
    r = client.post("/v1/chat/completions", headers=auth, json={
        "messages": [{"role": "user", "content": "weather?"}], "tools": TOOLS})
    assert r.status_code == 200
    prompt = seen[0].prompt
    # user text first, instructions at the end (ordered for the live agent)
    assert prompt.index("weather?") < prompt.index("If the request needs a tool")
    assert "only DECIDING" in prompt
    assert '"get_weather"' in prompt


def test_chat_tools_stream(client, auth, monkeypatch):
    _stub_chat(client, monkeypatch, CANNED)
    with client.stream("POST", "/v1/chat/completions", headers=auth, json={
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": TOOLS, "stream": True,
    }) as r:
        lines = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    chunks = [json.loads(x) for x in lines[:-1]]
    assert lines[-1] == "[DONE]"
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    text = "".join(d.get("content", "") for d in deltas)
    assert text == ""  # the bare JSON never leaks as content
    call_deltas = [d["tool_calls"] for d in deltas if "tool_calls" in d]
    assert len(call_deltas) == 1
    assert call_deltas[0][0]["function"]["name"] == "get_weather"
    assert call_deltas[0][0]["index"] == 0
    finishes = [c["choices"][0]["finish_reason"] for c in chunks
                if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finishes == ["tool_calls"]


def test_chat_tools_plain_reply_stream_untouched(client, auth, monkeypatch):
    _stub_chat(client, monkeypatch, "Just a normal answer.")
    with client.stream("POST", "/v1/chat/completions", headers=auth, json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS, "stream": True,
    }) as r:
        lines = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    chunks = [json.loads(x) for x in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "")
                   for c in chunks if c["choices"])
    assert text == "Just a normal answer."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_tools_swallowed_non_tool_json_recovered(client, auth, monkeypatch):
    _stub_chat(client, monkeypatch, '{"answer": "plain json data"}')
    with client.stream("POST", "/v1/chat/completions", headers=auth, json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS, "stream": True,
    }) as r:
        lines = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    chunks = [json.loads(x) for x in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "")
                   for c in chunks if c["choices"])
    assert text == '{"answer": "plain json data"}'  # flushed, not lost
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_tools_disabled_flag_passes_json_through(client, auth, monkeypatch):
    monkeypatch.setattr(client.app.state.services.settings, "tool_emulation", False)
    _stub_chat(client, monkeypatch, CANNED)
    r = client.post("/v1/chat/completions", headers=auth, json={
        "messages": [{"role": "user", "content": "weather?"}], "tools": TOOLS})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert '"tool_calls"' in choice["message"]["content"]


def test_chat_tool_choice_none_skips_injection(client, auth, monkeypatch):
    seen = _stub_chat(client, monkeypatch, CANNED)
    r = client.post("/v1/chat/completions", headers=auth, json={
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": TOOLS, "tool_choice": "none"})
    assert r.status_code == 200
    assert "If the request needs a tool" not in seen[0].prompt


def test_chat_history_roundtrip(client, auth, monkeypatch):
    seen = _stub_chat(client, monkeypatch, CANNED)
    r = client.post("/v1/chat/completions", headers=auth, json={
        "messages": [
            {"role": "user", "content": "weather in Beijing?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather",
                              "arguments": "{\"city\": \"Beijing\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 24C"},
        ],
        "tools": TOOLS})
    assert r.status_code == 200
    prompt = seen[0].prompt
    assert "Tool result from the caller" in prompt and "Sunny, 24C" in prompt
    assert '"get_weather"' in prompt  # the decided call is narrated as JSON


# ---------- integration: /v1/responses ----------

def test_responses_tools_non_stream(client, auth, monkeypatch):
    _stub_chat(client, monkeypatch, CANNED)
    r = client.post("/v1/responses", headers=auth, json={
        "input": "weather?", "tools": TOOLS})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    fc = [i for i in body["output"] if i["type"] == "function_call"]
    assert len(fc) == 1
    assert fc[0]["name"] == "get_weather"
    assert json.loads(fc[0]["arguments"]) == {"city": "Beijing"}
    assert fc[0]["call_id"].startswith("call_")


def test_responses_accepts_function_call_history(client, auth, monkeypatch):
    seen = _stub_chat(client, monkeypatch, CANNED)
    r = client.post("/v1/responses", headers=auth, json={
        "input": [
            {"role": "user", "content": "weather in Beijing?"},
            {"type": "function_call", "name": "get_weather",
             "arguments": "{\"city\": \"Beijing\"}", "call_id": "call_abc"},
            {"type": "function_call_output", "call_id": "call_abc",
             "output": "Sunny, 24C"},
        ],
        "tools": TOOLS})
    assert r.status_code == 200
    prompt = seen[0].prompt
    assert "Tool result from the caller" in prompt and "Sunny, 24C" in prompt


def test_responses_tools_stream(client, auth, monkeypatch):
    _stub_chat(client, monkeypatch, CANNED)
    with client.stream("POST", "/v1/responses", headers=auth, json={
        "input": "weather?", "tools": TOOLS, "stream": True}) as r:
        events = [json.loads(ln[6:]) for ln in r.iter_lines() if ln.startswith("data: ")]
    types = [e["type"] for e in events]
    assert types[0] == "response.created"
    assert "response.function_call_arguments.delta" in types
    assert types[-1] == "response.completed"
    seqs = [e["sequence_number"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    final = events[-1]["response"]
    fc = [i for i in final["output"] if i["type"] == "function_call"]
    assert len(fc) == 1 and fc[0]["name"] == "get_weather"
    text = "".join(e["delta"] for e in events
                   if e["type"] == "response.output_text.delta")
    assert text == ""  # bare JSON reply: nothing to stream as text
