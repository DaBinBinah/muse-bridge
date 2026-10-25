from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...core.media import load_image_ref
from ...core.models import resolve_model
from ...core.prompt import flatten_messages, message_turns
from ...core.tools import ToolStreamFilter, parse_tool_calls, tool_instructions
from ...drivers.base import ChatRequest, InputImage
from ...errors import InvalidRequest, Muse2APIError
from ...services.container import Services
from ..deps import get_services, require_api_key
from ..schemas import ChatCompletionRequest

log = logging.getLogger(__name__)
router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _sse(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n".encode()


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request,
                           svc: Services = Depends(get_services)):
    request.state.model = body.model  # logged even if it does not resolve
    spec = resolve_model(body.model, "chat")
    request.state.model = body.model or spec.id
    flat = flatten_messages([m.model_dump() for m in body.messages])
    if not flat.text and not flat.images:
        raise InvalidRequest("messages contain no text or images")

    # Emulated tool calling (C3): describe the tools in the prompt and parse the
    # marker format back out of the reply (see core/tools.py).
    tools_active = bool(svc.settings.tool_emulation and body.tools
                        and body.tool_choice != "none")
    prompt_text = ((tool_instructions(body.tools, body.tool_choice) + "\n\n" + flat.text)
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
        turns=message_turns([m.model_dump() for m in body.messages]),
        cancel=cancel,
    )
    completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    model_name = body.model or spec.id
    stream = svc.gateway.chat_stream(req)

    if not body.stream:
        text = "".join([d async for d in stream])
        message: dict = {"role": "assistant", "content": text}
        finish = "stop"
        if tools_active:
            content, calls = parse_tool_calls(text)
            if calls:
                message = {"role": "assistant", "content": content, "tool_calls": calls}
                finish = "tool_calls"
            elif content is not None:
                message["content"] = content
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": _approx_tokens(prompt_text),
                "completion_tokens": _approx_tokens(text),
                "total_tokens": _approx_tokens(prompt_text) + _approx_tokens(text),
            },
        }

    def chunk(delta: dict, finish: str | None = None) -> dict:
        return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    async def events(first: str | None) -> AsyncIterator[bytes]:
        # ``first`` is the pre-fetched first delta when heartbeats are disabled;
        # when enabled it is None and the first token is awaited here, after the
        # 200 (headers) has already been committed to the client.
        errored = False
        try:
            if first is None and svc.settings.stream_heartbeat_interval > 0:
                # C4: some clients/proxies drop idle SSE connections while the
                # gateway waits for the first upstream token (up to
                # first_token_timeout). Send SSE comment heartbeats until it
                # arrives. asyncio.wait (not wait_for) polls so the in-flight
                # __anext__ is never cancelled between heartbeats.
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
            produced = first or ""
            if tools_active:
                # Stream through the marker gate: text is passed through (with a
                # partial-tag holdback) until a <tool_call> appears, then the
                # rest is swallowed and returned as tool_calls deltas instead.
                filt = ToolStreamFilter()
                out = filt.feed(produced)
                yield _sse(chunk({"role": "assistant", "content": out}))
                async for delta in stream:
                    produced += delta
                    out = filt.feed(delta)
                    if out:
                        yield _sse(chunk({"content": out}))
                out = filt.finish()
                if out:
                    yield _sse(chunk({"content": out}))
                _, calls = parse_tool_calls(produced)
                if calls:
                    for i, call in enumerate(calls):
                        yield _sse(chunk({"tool_calls": [dict(call, index=i)]}))
                    yield _sse(chunk({}, "tool_calls"))
                else:
                    yield _sse(chunk({}, "stop"))
            else:
                yield _sse(chunk({"role": "assistant", "content": produced}))
                async for delta in stream:
                    produced += delta
                    yield _sse(chunk({"content": delta}))
                yield _sse(chunk({}, "stop"))
            if (body.stream_options or {}).get("include_usage"):
                usage = {"prompt_tokens": _approx_tokens(prompt_text),
                         "completion_tokens": _approx_tokens(produced)}
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                yield _sse({"id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name, "choices": [],
                            "usage": usage})
        except Muse2APIError as exc:
            errored = True
            yield _sse(exc.to_body())
        except Exception:  # noqa: BLE001
            errored = True
            log.exception("stream crashed")
            yield _sse({"error": {"message": "internal error", "type": "server_error"}})
        finally:
            cancel.set()
            await stream.aclose()
        # [DONE] marks a well-formed stream; after an error event some clients
        # would treat it as a normal completion and ignore the error.
        if not errored:
            yield _sse("[DONE]")

    # With stream_heartbeat_interval set (C4), the 200 goes out immediately and
    # ': keepalive' SSE comments are sent while the first token is pending;
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
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
