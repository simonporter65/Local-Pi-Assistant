"""
routes/chat.py — Chat endpoint and model streaming.

POST /chat      — stream a chat response
POST /chat/stop — stop streaming and resume heartbeat
"""

import asyncio
import json
import re
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from core.fast_classifier import fast_classify
from core.log import get_logger
from core.router import route_to_model
from core.token_budget import get_token_budget
from state import AppState

logger = get_logger("chat")
router = APIRouter()

# ── Session history (in-process; keyed by session_id) ────────────────────────

_session_histories: dict = {}
_session_last_used: dict = {}
MAX_HISTORY  = 6
MAX_SESSIONS = 50
SESSION_TTL  = 3600  # seconds before idle session is pruned


def _get_history(session_id: str) -> list:
    _session_last_used[session_id] = time.time()
    return _session_histories.setdefault(session_id, [])


def _add_to_history(session_id: str, role: str, msg: str):
    h = _get_history(session_id)
    h.append({"role": role, "content": msg})
    if len(h) > MAX_HISTORY * 2:
        del h[:-MAX_HISTORY * 2]
    # Prune stale sessions if over the cap
    if len(_session_histories) > MAX_SESSIONS:
        now = time.time()
        stale = [sid for sid, ts in _session_last_used.items() if now - ts > SESSION_TTL]
        for sid in stale:
            _session_histories.pop(sid, None)
            _session_last_used.pop(sid, None)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: Request):
    arc = request.app.state.arc
    body = await request.json()
    msg = body.get("message", "").strip()
    if not msg:
        return {"error": "empty"}
    session_id = request.headers.get("X-Session-ID", "default")
    arc.heartbeat.pause_for_user()
    return StreamingResponse(
        _chat_stream(msg, session_id, arc),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/stop")
async def chat_stop(request: Request):
    """Called by the UI when the user hits the stop button."""
    arc = request.app.state.arc
    arc.heartbeat.resume_after_user()
    return {"ok": True}


# ── Chat stream ───────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _chat_stream(
    user_message: str,
    session_id: str,
    arc: AppState,
) -> AsyncGenerator[str, None]:

    def sse(t, **kw):
        logger.debug("SSE %s: %s", t, kw)
        return _sse({"type": t, **kw})

    t0 = time.time()
    name = arc.personality.name or "Assistant"

    # Classify immediately (~0ms heuristic)
    _add_to_history(session_id, "user", user_message)
    intent   = fast_classify(user_message)
    category = intent.get("category", "general_chat")
    logger.info("classify [%s]: %s → %s", intent["_source"], user_message[:50], category)

    # Get user context in background thread (cached 30s)
    user_ctx = await asyncio.to_thread(arc.user_model.get_context_for_prompt)
    asyncio.create_task(asyncio.to_thread(arc.training.score_previous_exchange, user_message, session_id))
    asyncio.create_task(asyncio.to_thread(arc.user_model.extract_from_message, user_message))
    asyncio.create_task(arc.broadcast({"type": "profile_updated"}))

    route    = route_to_model(intent)
    model    = route["model"]
    latency  = route.get("latency", "fast")
    thinking = route.get("thinking", False)
    budget   = get_token_budget(latency, category)

    yield sse("model", name=model, latency=latency, category=category, thinking=thinking)
    await asyncio.sleep(0)

    rewritten = user_message
    # Only search memory if message explicitly references past context
    memory_triggers = {"remember", "earlier", "last time", "you said", "we discussed",
                       "before", "previously", "again", "still", "anymore"}
    if set(user_message.lower().split()) & memory_triggers:
        past = await asyncio.to_thread(arc.memory.semantic_search, user_message, 3)
        past_ctx = "\n".join([
            f"- '{p['input'][:50]}' → '{p['output'][:80]}'"
            for p in past
        ]) or "None yet."
    else:
        past_ctx = "None yet."

    # Skills list: agentic categories → all skills; personality_change → set_personality only;
    # other categories → custom (agent-written) skills only.
    _AGENTIC_SKILL_CATS = {
        "research", "coding", "debugging", "agentic_task", "web_browsing",
        "data_analysis", "file_management", "shell_command",
        "screenshot_analysis", "image_description",
    }
    custom_skills = arc.registry.list_custom_skills()
    if category in _AGENTIC_SKILL_CATS:
        skills_list = arc.registry.list_skills()
    elif category == "personality_change":
        skills_list = json.dumps(
            {"set_personality": arc.registry.skills["set_personality"].DESCRIPTION},
            indent=2,
        ) if "set_personality" in arc.registry.skills else ""
    elif custom_skills:
        skills_list = custom_skills
    else:
        skills_list = ""

    system = arc.personality.get_full_system_prompt(model, category, user_ctx, past_ctx, skills_list)

    # Web search — pre-execute and inject results (more reliable than SKILL: format)
    if category == "web_search":
        yield sse("stage", message="Searching the web...")
        await asyncio.sleep(0)
        try:
            search_results = await asyncio.wait_for(
                asyncio.to_thread(arc.registry.run, "web_search", query=rewritten, max_results=5),
                timeout=20.0,
            )
            injected = str(search_results)[:6000]
            system += f"\n\nWEB SEARCH RESULTS for '{rewritten}':\n{injected}\n\nSynthesize these results into a helpful, accurate response."
            logger.info("web_search pre-executed for: %s", rewritten[:50])
        except asyncio.TimeoutError:
            logger.warning("web_search pre-execute timed out after 20s")
        except Exception as e:
            logger.warning("web_search pre-execute failed: %s", e)

    # Web browsing with a direct URL — pre-fetch page content
    if category == "web_browsing":
        url_match = re.search(r'https?://[^\s"\'<>]+', user_message)
        if url_match:
            found_url = url_match.group(0)
            interaction_words = {"click", "fill", "submit", "login", "sign", "type", "scroll"}
            if not set(user_message.lower().split()) & interaction_words:
                yield sse("stage", message="Fetching page...")
                await asyncio.sleep(0)
                try:
                    page_content = await asyncio.wait_for(
                        asyncio.to_thread(arc.registry.run, "web_fetch", url=found_url, max_chars=4000),
                        timeout=20.0,
                    )
                    system += f"\n\nPAGE CONTENT from {found_url}:\n{str(page_content)[:6000]}\n\nAnswer based on this content."
                    logger.info("web_fetch pre-executed for: %s", found_url[:60])
                except asyncio.TimeoutError:
                    logger.warning("web_fetch pre-execute timed out for: %s", found_url[:60])
                except Exception as e:
                    logger.warning("web_fetch pre-execute failed: %s", e)

    # Image/screenshot — pre-run vision skill
    if category in {"screenshot_analysis", "image_description"}:
        img_match = re.search(r'(/[^\s"\']+\.(?:png|jpg|jpeg|gif|webp))', user_message, re.I)
        if img_match:
            img_path = img_match.group(1)
            yield sse("stage", message="Analyzing image...")
            await asyncio.sleep(0)
            try:
                question = user_message.replace(img_path, "").strip() or "Describe what you see in detail."
                vision_result = await asyncio.wait_for(
                    asyncio.to_thread(arc.registry.run, "vision", image_path=img_path, question=question),
                    timeout=30.0,
                )
                system += f"\n\nIMAGE ANALYSIS of {img_path}:\n{str(vision_result)[:6000]}\n\nAnswer based on this analysis."
                logger.info("vision pre-executed for: %s", img_path)
            except asyncio.TimeoutError:
                logger.warning("vision pre-execute timed out for: %s", img_path)
            except Exception as e:
                logger.warning("vision pre-execute failed: %s", e)

    yield sse("stage", message=f"{name} is thinking...")
    await asyncio.sleep(0)

    final = "Something went wrong — please try again."
    result = None

    use_skills = (
        category in _AGENTIC_SKILL_CATS
        or category == "personality_change"
        or bool(custom_skills)
    )

    try:
        async for event_type, data in _run_model_streaming(
            rewritten, model, system, budget,
            _get_history(session_id),
            use_skills=use_skills,
            use_thinking=thinking,
            registry=arc.registry,
        ):
            if event_type == "token":
                yield sse("token", text=data)
            elif event_type in ("think", "think_token"):
                yield sse("thinking_token", text=data)
            elif event_type == "skill":
                yield sse("skill_call", skill=data.get("name", "?"), args=data.get("args", {}))
            elif event_type == "done":
                result = data
                final = data.get("output", final)
                break
    except asyncio.TimeoutError:
        logger.warning("Chat model timed out")
        final = "Sorry, that took too long. Please try again."
    except Exception as e:
        logger.error("Chat error: %s: %s", type(e).__name__, e, exc_info=True)
        final = "Something went wrong — please try again."

    # If personality was changed, hot-reload config so next message uses new settings
    if category == "personality_change":
        arc.personality._load()
        yield sse("personality_updated", name=arc.personality.name or "Assistant")

    # Track response in history
    if final and final != "Something went wrong — please try again.":
        _add_to_history(session_id, "assistant", final)

    try:
        yield sse("stage_done", message="Done")
        yield sse("final", message=final)

        asyncio.create_task(_post_response(
            user_message, final, intent, result, model, session_id, t0, category, arc,
        ))
    finally:
        arc.heartbeat.resume_after_user()


async def _post_response(user_message, final, intent, result, model, session_id, t0, category, arc: AppState):
    """All post-response work — fires as a background task after final is sent."""
    dur = int((time.time() - t0) * 1000)
    if not final or "went wrong" in final:
        await asyncio.to_thread(
            arc.memory.log_interaction, user_message, intent,
            (result or {}).get("model", model), final,
            False, (result or {}).get("tool_calls", 0), dur,
        )
        return

    sys_prompt = arc.personality.get_full_system_prompt(model, category, "", "")

    await asyncio.gather(
        asyncio.to_thread(
            arc.memory.log_interaction, user_message, intent,
            (result or {}).get("model", model), final,
            (result or {}).get("success", True),
            (result or {}).get("tool_calls", 0), dur,
        ),
        asyncio.to_thread(arc.user_model.extract_from_exchange, user_message, final),
        asyncio.to_thread(arc.training.record_exchange, sys_prompt, user_message, final, session_id, model),
    )
    await arc.broadcast({"type": "profile_updated"})

    pro = await asyncio.to_thread(arc.proactive.check_after_message, user_message, final)
    if pro:
        await arc.broadcast({"type": "proactive", "message": pro})

    if category in ("research", "web_search", "planning", "agentic_task", "coding"):
        summary = await asyncio.to_thread(arc.task_queue.summary)
        if summary.get("pending", 0) < 10:
            await asyncio.to_thread(
                arc.task_queue.add,
                title=f"Follow up: {user_message[:55]}",
                description=(
                    f"User asked: {user_message}\n"
                    f"Response: {final[:300]}\n\n"
                    f"Dig deeper. Find additional useful info. Prepare a proactive update."
                ),
                task_type="research",
                priority_name="low",
            )


# ── Model streaming ───────────────────────────────────────────────────────────

async def _iter_stream(stream):
    """Wrap synchronous ollama stream iterator for async use."""
    for chunk in stream:
        yield chunk
        await asyncio.sleep(0)


def _needed_ctx(system: str, messages: list, budget: int) -> int:
    """Estimate num_ctx needed and snap to a power-of-2 level.

    Smaller num_ctx = smaller KV cache = faster prefill + less RAM.
    We cap at 8192 to avoid requesting more than any model supports.
    """
    total_chars = len(system) + sum(len(m.get("content", "")) for m in messages)
    input_tokens = total_chars // 4 + 64
    needed = min(input_tokens + budget, 8192)
    for level in (1024, 2048, 4096, 8192):
        if needed <= level:
            return level
    return 8192


async def _run_model_streaming(
    prompt, model, system, budget, history=None,
    use_skills=False, use_thinking=False, registry=None,
):
    """Stream tokens from the model.

    Yields:
      ("token",       text)   — content token to stream to user
      ("think_token", text)   — thinking token (display as grey reasoning text)
      ("skill",       dict)   — skill call to execute
      ("done",        dict)   — final result {output, success, tool_calls, model}
    """
    import ollama

    think_kwargs = {} if use_thinking else {"think": False}

    if use_skills:
        user_content = f"Task: {prompt}\n\nUse SKILL: {{...}} or FINAL: <answer>"
    else:
        user_content = prompt

    messages = list(history or [])
    if messages and messages[-1]["role"] == "user":
        messages[-1] = {"role": "user", "content": user_content}
    else:
        messages.append({"role": "user", "content": user_content})

    tool_count = 0
    stuck_count = 0
    last_reply = ""
    first_call = True
    token_buffer = ""

    while tool_count < 20:
        msgs_with_system = [{"role": "system", "content": system}] + messages

        if first_call:
            first_call = False
            num_ctx = _needed_ctx(system, msgs_with_system, budget)
            stream = await asyncio.to_thread(
                ollama.chat, model=model, messages=msgs_with_system,
                stream=True,
                options={"temperature": 0.7, "num_predict": budget, "num_ctx": num_ctx},
                **think_kwargs,
            )
            collected = []
            streaming_started = not use_skills
            async for chunk in _iter_stream(stream):
                msg = chunk.get("message", {})
                think_tok = msg.get("thinking") or ""
                if think_tok:
                    yield ("think_token", think_tok)
                token = msg.get("content") or ""
                if not token:
                    continue
                collected.append(token)
                token_buffer += token
                if not streaming_started:
                    if "FINAL:" in token_buffer:
                        token_buffer = token_buffer.split("FINAL:", 1)[1].lstrip()
                        streaming_started = True
                        if token_buffer:
                            yield ("token", token_buffer)
                    elif token_buffer.lstrip().startswith("SKILL:") or token_buffer.lstrip().startswith("{\""):
                        pass  # Buffering a tool call silently
                    elif len(token_buffer) > 40:
                        streaming_started = True
                        yield ("token", token_buffer)
                else:
                    yield ("token", token)
            raw = "".join(collected)
        else:
            num_ctx = _needed_ctx(system, msgs_with_system, budget)
            resp = await asyncio.to_thread(
                ollama.chat, model=model, messages=msgs_with_system,
                options={"temperature": 0.7, "num_predict": budget, "num_ctx": num_ctx},
                **think_kwargs,
            )
            raw = resp["message"].get("content") or ""
            think_text = resp["message"].get("thinking") or ""
            if think_text:
                yield ("think_token", think_text)

        last_reply = raw
        # Strip any legacy <think> blocks
        reply = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        messages.append({"role": "assistant", "content": raw})

        final_m = re.search(r"FINAL:[\s\n]*(.*)", reply, re.DOTALL)
        if final_m:
            yield ("done", {"output": final_m.group(1).strip(), "success": True,
                            "tool_calls": tool_count, "model": model})
            return

        skill_m = re.search(r"SKILL:\s*(\{.*?\})", reply, re.DOTALL)
        if skill_m and registry:
            try:
                _decoder = json.JSONDecoder()
                sc, _ = _decoder.raw_decode(skill_m.group(1))
                yield ("skill", sc)
                res = await asyncio.to_thread(registry.run, sc["name"], **sc.get("args", {}))
                res_str = str(res)[:6000]
                messages.append({"role": "user", "content": f"Skill result:\n{res_str}\n\nContinue."})
                tool_count += 1
                continue
            except Exception as e:
                messages.append({"role": "user", "content": f"Skill error: {e}. Try another."})
                tool_count += 1
                continue

        if last_reply.strip():
            yield ("done", {"output": last_reply.strip(), "success": True,
                            "tool_calls": tool_count, "model": model})
            return

        stuck_count += 1
        if stuck_count >= 3:
            yield ("done", {"output": "I'm sorry, I had trouble generating a response. Please try again.",
                            "success": False, "tool_calls": tool_count, "model": model})
            return
        messages.append({"role": "user", "content": "Continue. SKILL: or FINAL: required."})

    yield ("done", {"output": last_reply.strip() or "Something went wrong — please try again.",
                    "success": bool(last_reply.strip()), "tool_calls": tool_count, "model": model})
