"""OpenAI Responses API (``/v1/responses``).

Maps ``input`` / ``instructions`` onto ``flatten_messages`` and reuses
``Gateway.chat_stream``. Streaming emits the standard Responses events
(``response.created`` / ``response.output_text.delta`` / ``response.completed``).

Tool items (``function_call`` / ``function_call_output``) are translated into
the emulated marker format from ``core/tools.py`` and parsed back into
``function_call`` output items; ``reasoning`` items the SDK round-trips are
ignored. muse.ai still has no native function calling, so this is best-effort.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from ...core.media import load_image_ref
from ...core.models import resolve_model
from ...core.prompt import flatten_messages, message_turns
from ...core.tools import ToolStreamFilter, compose_tool_prompt, parse_tool_calls
from ...drivers.base import ChatRequest, InputImage
from ...errors import InvalidRequest, Muse2APIError
from ...services.container import Services
from ..deps import get_services, require_api_key

log = logging.getLogger(__name__)
router = APIRouter(tags=["responses"], dependencies=[Depends(require_api_key)])


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _sse(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n".encode()


class ResponseRequest(BaseModel):
    model: str | None = None
    input: Any = None
    instructions: str | None = None
    stream: bool = False
    user: str | None = None
    max_output_tokens: int | None = None

    model_config = ConfigDict(extra="allow")


# Item types that cannot be emulated (only function calling can).
_UNSUPPORTED_ITEMS = {"computer_call", "file_search_call", "web_search_call"}


def _to_messages(body: ResponseRequest) -> list[dict]:
    """Convert Responses ``input`` / ``instructions`` to chat-style messages."""
    messages: list[dict] = []
    if body.instructions:
        messages.append({"role": "system", "content": body.instructions})
    inp = body.input
    if inp is None or (isinstance(inp, str) and not inp.strip()):
        pass
    elif isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                raise InvalidRequest("input items must be objects")
            itype = item.get("type")
            if itype == "function_call":
                messages.append({"role": "assistant", "content": None, "tool_calls": [{
                    "id": item.get("call_id") or item.get("id") or "call_0",
                    "type": "function",
                    "function": {"name": item.get("name") or "",
                                 "arguments": item.get("arguments") or "{}"}}]})
            elif itype == "function_call_output":
                out = item.get("output")
                if isinstance(out, dict):
                    out = out.get("content") or json.dumps(out, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                                 "content": out if isinstance(out, str) else ""})
            elif itype == "reasoning":
                continue  # the SDK round-trips its own reasoning items; ignore
            elif itype in _UNSUPPORTED_ITEMS:
                raise InvalidRequest(f"input item type '{itype}' is not supported")
            elif "role" in item or itype in ("message", None):
                messages.append({"role": item.get("role", "user"),
                                 "content": item.get("content")})
            else:
                raise InvalidRequest(f"unsupported input item type '{itype}'")
    else:
        raise InvalidRequest("input must be a string or a list of items")
    return messages


def _message_item(msg_id: str, text: str) -> dict:
    return {"id": msg_id, "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "annotations": [], "text": text}]}


def _function_call_item(call: dict) -> dict:
    return {"id": "fc_" + uuid.uuid4().hex[:24], "type": "function_call",
            "status": "completed", "call_id": call["id"],
            "name": call["function"]["name"], "arguments": call["function"]["arguments"]}


def _usage(input_text: str, output_text: str) -> dict:
    i, o = _approx_tokens(input_text), _approx_tokens(output_text)
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0}}


def _response_object(resp_id: str, msg_id: str, created: int, model: str, text: str,
                     input_text: str, *, status: str = "completed",
                     output: list[dict] | None = None, tools: list | None = None) -> dict:
    return {"id": resp_id, "object": "response", "created_at": created, "model": model,
            "status": status,
            "output": (output if output is not None
                       else ([_message_item(msg_id, text)] if status == "completed" else [])),
            "usage": _usage(input_text, text), "parallel_tool_calls": True,
            "tool_choice": "auto", "tools": tools or []}


@router.post("/v1/responses")
async def create_response(body: ResponseRequest, request: Request,
                          svc: Services = Depends(get_services)):
    request.state.model = body.model  # logged even if it does not resolve
    spec = resolve_model(body.model, "chat")
    request.state.model = body.model or spec.id
    messages = _to_messages(body)
    flat = flatten_messages(messages)
    if not flat.text and not flat.images:
        raise InvalidRequest("input contains no text or images")
    turns = message_turns(messages)

    # Emulated tool calling (C3), same as chat completions: headerless narration.
    req_tools = getattr(body, "tools", None) or []
    tools_active = (bool(svc.settings.tool_emulation and req_tools)
                    and getattr(body, "tool_choice", None) != "none")
    prompt_text = (compose_tool_prompt(turns, req_tools,
                                       getattr(body, "tool_choice", None))
                   if tools_active else flat.text)

    images = [InputImage(*(await load_image_ref(ref))) for ref in flat.images]
    cancel = asyncio.Event()
    req = ChatRequest(
        prompt=prompt_text,
        model=spec.id,
        images=images,
        timeout=svc.settings.chat_timeout,
        first_token_timeout=svc.settings.first_token_timeout,
        conversation_hint=body.user,
        turns=turns,
        cancel=cancel,
    )
    stream = svc.gateway.chat_stream(req)
    resp_id = "resp_" + uuid.uuid4().hex[:24]
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    created = int(time.time())
    model_name = body.model or spec.id

    if not body.stream:
        text = "".join([d async for d in stream])
        if tools_active:
            content, calls = parse_tool_calls(text)
            if calls:
                items = ([_message_item(msg_id, content)] if content else []) + [
                    _function_call_item(c) for c in calls]
                return _response_object(resp_id, msg_id, created, model_name, text,
                                        prompt_text, output=items, tools=req_tools)
            out_text = content if content is not None else text
            return _response_object(resp_id, msg_id, created, model_name, out_text,
                                    prompt_text, tools=req_tools)
        return _response_object(resp_id, msg_id, created, model_name, text, prompt_text)

    async def events(first: str | None) -> AsyncIterator[bytes]:
        # ``first`` is the pre-fetched first delta when heartbeats are disabled;
        # when enabled it is None and the first token is awaited here, after the
        # 200 (headers) has already been committed to the client.
        seq = 0
        try:
            if first is None and svc.settings.stream_heartbeat_interval > 0:
                # Same C4 pattern as chat completions: send SSE comment
                # heartbeats while the gateway waits for the first token.
                pending = asyncio.ensure_future(anext(stream))
                try:
                    while True:
                        done, _ = await asyncio.wait(
                            {pending},
                            timeout=svc.settings.stream_heartbeat_interval)
                        if done:
                            try:
                                first = pending.result()
                            except StopAsyncIteration:
                                first = ""
                            break
                        yield b": keepalive\n\n"
                finally:
                    if not pending.done():
                        pending.cancel()
            yield _sse({"type": "response.created", "sequence_number": seq,
                        "response": _response_object(resp_id, msg_id, created, model_name,
                                                     "", prompt_text, status="in_progress")})
            seq += 1
            # The official SDK builds its accumulated snapshot from item/part
            # lifecycle events, so emit them around the text deltas.
            yield _sse({"type": "response.in_progress", "sequence_number": seq,
                        "response": _response_object(resp_id, msg_id, created, model_name,
                                                     "", prompt_text, status="in_progress")})
            seq += 1
            yield _sse({"type": "response.output_item.added", "sequence_number": seq,
                        "output_index": 0,
                        "item": {"id": msg_id, "type": "message", "status": "in_progress",
                                 "role": "assistant", "content": []}})
            seq += 1
            yield _sse({"type": "response.content_part.added", "sequence_number": seq,
                        "item_id": msg_id, "output_index": 0, "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []}})
            seq += 1
            produced = first or ""
            filt = ToolStreamFilter() if tools_active else None

            def delta_event(delta: str) -> dict:
                return {"type": "response.output_text.delta", "sequence_number": seq,
                        "item_id": msg_id, "output_index": 0, "content_index": 0,
                        "delta": delta, "logprobs": []}

            def safe(delta: str) -> str:
                return filt.feed(delta) if filt else delta

            out = safe(produced)
            if out:
                yield _sse(delta_event(out))
                seq += 1
            async for delta in stream:
                produced += delta
                out = safe(delta)
                if out:
                    yield _sse(delta_event(out))
                    seq += 1
            if filt:
                pending = filt.finish()
            else:
                pending = ""

            content, calls = (parse_tool_calls(produced) if tools_active else (produced, []))
            if not calls and pending:
                # A swallowed payload that did not parse into calls is still
                # legitimate content (e.g. the model just answered in JSON).
                yield _sse(delta_event(pending))
                seq += 1
            final_output: list[dict] = []
            if calls:
                # Close the text message item first; with the bare-JSON protocol
                # it is usually empty.
                yield _sse({"type": "response.output_item.done", "sequence_number": seq,
                            "output_index": 0, "item": _message_item(msg_id, content or "")})
                seq += 1
                if content:
                    final_output.append(_message_item(msg_id, content))
                for i, call in enumerate(calls, start=1):
                    item = _function_call_item(call)
                    yield _sse({"type": "response.output_item.added", "sequence_number": seq,
                                "output_index": i, "item": {**item, "status": "in_progress"}})
                    seq += 1
                    yield _sse({"type": "response.function_call_arguments.delta",
                                "sequence_number": seq, "item_id": item["id"],
                                "output_index": i, "delta": item["arguments"]})
                    seq += 1
                    yield _sse({"type": "response.output_item.done", "sequence_number": seq,
                                "output_index": i, "item": item})
                    seq += 1
                    final_output.append(item)
            else:
                yield _sse({"type": "response.output_item.done", "sequence_number": seq,
                            "output_index": 0, "item": _message_item(msg_id, produced)})
                seq += 1
                final_output.append(_message_item(msg_id, produced))
            yield _sse({"type": "response.completed", "sequence_number": seq,
                        "response": _response_object(resp_id, msg_id, created, model_name,
                                                     produced, prompt_text,
                                                     output=final_output, tools=req_tools)})
        except Muse2APIError as exc:
            yield _sse(exc.to_body())
        except Exception:  # noqa: BLE001
            log.exception("responses stream crashed")
            yield _sse({"error": {"message": "internal error", "type": "server_error"}})
        finally:
            cancel.set()
            await stream.aclose()

    # With stream_heartbeat_interval set (C4), the 200 goes out immediately;
    # upstream errors then surface as SSE error events instead of HTTP codes.
    if svc.settings.stream_heartbeat_interval > 0:
        return StreamingResponse(events(None), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # Default: pull the first delta before committing to a 200 so that
    # account/upstream failures surface as proper HTTP errors instead of a
    # broken event stream.
    try:
        first = await anext(stream)
    except StopAsyncIteration:
        first = ""

    return StreamingResponse(events(first), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
