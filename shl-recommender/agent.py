from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import google.generativeai as genai

import vector_store

_DIR = Path(__file__).resolve().parent
_CATALOG_PATH = Path(os.environ.get("SHL_CATALOG_PATH", str(_DIR / "catalog.json")))

SYSTEM_PROMPT = """You are an SHL assessment recommendation assistant. Your only job is to help
hiring managers and recruiters find the right SHL Individual Test assessments.

Rules:
1. If the query is vague, ask ONE clarifying question before recommending.
2. Only recommend assessments from the provided catalog context.
3. Never invent or hallucinate assessment names or URLs.
4. Refuse any question not related to SHL assessments.
5. When comparing assessments, use only the catalog descriptions provided.
6. Keep responses concise and professional.
7. Maximum 8 turns per conversation.

Catalog context will be injected below each user message.

You MUST respond with a single JSON object only (no markdown), shape:
{"reply": string, "recommendations": [{"name": string, "url": string, "test_type": string}], "end_of_conversation": boolean}

When still clarifying, recommendations must be [].
Every recommendation must use a url copied exactly from the provided catalog context."""

MAX_TURNS = 8
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_INJECTION = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions|disregard\s+(the\s+)?(above|prior)|"
    r"you\s+are\s+now\s+(a\s+)?(DAN|developer)|new\s+instructions\s*:|"
    r"system\s*:\s*override|jailbreak|reveal\s+(your\s+)?prompt)",
    re.I,
)
_LEGAL = re.compile(
    r"\b(lawsuit|sue\s|legal\s+advice|lawyer|attorney|contract\s+law|"
    r"discrimination\s+claim|gdpr\s+complaint)\b",
    re.I,
)
_CONTEXT_HINTS = re.compile(
    r"\b(java|python|developer|engineer|software|sales|manager|graduate|intern|"
    r"cognitive|ability|aptitude|numerical|verbal|personality|opq|verify|"
    r"remote|adaptive|skill|coding|role|senior|mid|junior|lead|hire|team|"
    r"customer|analyst|finance|marketing|warehouse|technical|behavior)\b",
    re.I,
)


@lru_cache(maxsize=1)
def _load_catalog() -> list[dict[str, Any]]:
    if not _CATALOG_PATH.is_file():
        return []
    try:
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _catalog_url_set(catalog: list[dict[str, Any]]) -> set[str]:
    return {x.get("url", "") for x in catalog if isinstance(x, dict) and x.get("url")}


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        out.append({"role": role, "content": content.strip()})
    return out


def _user_turns(messages: list[dict[str, str]]) -> int:
    return sum(1 for m in messages if m["role"] == "user")


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def _conversation_tail(messages: list[dict[str, str]], max_chars: int = 3500) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(f'{m["role"].upper()}: {m["content"]}')
    text = "\n".join(parts)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _is_prompt_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))


def _is_legal(text: str) -> bool:
    return bool(_LEGAL.search(text or ""))


def _is_off_topic_non_shl(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False
    if "shl" in t or "assessment" in t or "test" in t or "opq" in t or "verify" in t:
        return False
    interview_only = re.search(
        r"\b(best|good|favorite)\s+(interview|phone screen)\s+question",
        t,
    )
    if interview_only and "assessment" not in t and "test" not in t:
        return True
    return False


def _is_vague_first_turn(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return True
    vague_phrases = (
        "i need an assessment",
        "need an assessment",
        "help me hire",
        "looking for an assessment",
        "recommend an assessment",
        "which assessment",
        "what assessment should i use",
    )
    if any(p in t for p in vague_phrases) and not _CONTEXT_HINTS.search(t):
        return True
    if len(t.split()) <= 5 and not _CONTEXT_HINTS.search(t):
        return True
    return False


def _parse_compare_names(text: str) -> tuple[str, str] | None:
    m = re.search(
        r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        text,
        re.I | re.S,
    )
    if not m:
        m = re.search(r"compare\s+(.+?)\s+to\s+(.+?)(?:\?|$)", text, re.I | re.S)
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    a = re.sub(r'^["\']|["\']$', "", a).strip()
    b = re.sub(r'^["\']|["\']$', "", b).strip()
    if len(a) < 2 or len(b) < 2:
        return None
    return a, b


def _find_by_hint(catalog: list[dict[str, Any]], hint: str) -> dict[str, Any] | None:
    hint_l = hint.lower().strip()
    best: dict[str, Any] | None = None
    best_score = 0
    for item in catalog:
        name = (item.get("name") or "").lower()
        if hint_l == name:
            return item
        if hint_l in name or name in hint_l:
            score = min(len(hint_l), len(name))
            if score > best_score:
                best_score = score
                best = item
        elif all(tok in name for tok in hint_l.split() if len(tok) > 2):
            score = len(hint_l)
            if score > best_score:
                best_score = score
                best = item
    return best


def _catalog_compare_block(catalog: list[dict[str, Any]], last_user: str) -> str | None:
    parsed = _parse_compare_names(last_user)
    if not parsed:
        return None
    a, b = parsed
    ia = _find_by_hint(catalog, a)
    ib = _find_by_hint(catalog, b)
    if not ia or not ib:
        return None
    lines = [
        "COMPARISON (catalog facts only — use these verbatim facts, do not invent):",
        f"Assessment A: {ia.get('name','')}",
        f"  URL: {ia.get('url','')}",
        f"  test_type: {ia.get('test_type','')}",
        f"  remote_testing: {ia.get('remote_testing')}",
        f"  adaptive: {ia.get('adaptive')}",
        f"  duration: {ia.get('duration','')}",
        f"  description: {ia.get('description','')}",
        f"Assessment B: {ib.get('name','')}",
        f"  URL: {ib.get('url','')}",
        f"  test_type: {ib.get('test_type','')}",
        f"  remote_testing: {ib.get('remote_testing')}",
        f"  adaptive: {ib.get('adaptive')}",
        f"  duration: {ib.get('duration','')}",
        f"  description: {ib.get('description','')}",
    ]
    return "\n".join(lines)


def _search_context_block(query: str, top_k: int = 12) -> str:
    try:
        hits = vector_store.search(query, top_k=top_k)
    except Exception:
        hits = []
    if not hits:
        return "CATALOG_SEARCH_RESULTS: (none — catalog empty or index missing)"
    lines = ["CATALOG_SEARCH_RESULTS (use only these for recommendations):"]
    for h in hits:
        lines.append(
            json.dumps(
                {
                    "name": h.get("name"),
                    "url": h.get("url"),
                    "test_type": h.get("test_type"),
                    "description": (h.get("description") or "")[:500],
                    "remote_testing": h.get("remote_testing"),
                    "adaptive": h.get("adaptive"),
                    "duration": h.get("duration"),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _sanitize_recommendations(
    recs: list[Any],
    allowed_urls: set[str],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(recs, list):
        return out
    for r in recs:
        if not isinstance(r, dict):
            continue
        url = str(r.get("url", "")).strip()
        if url not in allowed_urls:
            continue
        name = str(r.get("name", "")).strip()
        tt = str(r.get("test_type", "")).strip()
        out.append({"name": name, "url": url, "test_type": tt})
    return out[:10]


def _invoke_gemini_json(user_payload: str) -> dict[str, Any] | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    try:
        resp = model.generate_content(
            user_payload,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.35,
            ),
        )
        text = (resp.text or "").strip()
        return json.loads(text)
    except Exception:
        return None


def run_agent(messages: list[dict]) -> tuple[str, list[dict], bool]:
    catalog = _load_catalog()
    allowed = _catalog_url_set(catalog)
    msgs = _normalize_messages(messages)
    if not msgs:
        return (
            "Please send at least one message so I can help you choose SHL assessments.",
            [],
            False,
        )

    last_user = _last_user_text(msgs)
    turns = _user_turns(msgs)

    if _is_prompt_injection(last_user):
        return (
            "I cannot follow instructions that attempt to override my rules. "
            "I can only help with SHL assessment selection.",
            [],
            False,
        )
    if _is_legal(last_user):
        return (
            "I cannot provide legal guidance. I can only help with SHL assessment selection.",
            [],
            False,
        )
    if _is_off_topic_non_shl(last_user):
        return (
            "I can only help with SHL assessment selection.",
            [],
            False,
        )

    cmp_names = _parse_compare_names(last_user)
    if cmp_names:
        ia = _find_by_hint(catalog, cmp_names[0])
        ib = _find_by_hint(catalog, cmp_names[1])
        if not ia or not ib:
            return (
                "I could not find both assessments in the current SHL catalog. "
                "Please check the exact product names as listed on shl.com.",
                [],
                False,
            )
        cmp_block = _catalog_compare_block(catalog, last_user)
        if not cmp_block:
            return (
                "Unable to build a catalog-grounded comparison for this request.",
                [],
                False,
            )
        tail = _conversation_tail(msgs)
        payload = f"""CONVERSATION:\n{tail}\n\n{cmp_block}\n\nRespond with JSON only.
Compare the two assessments using ONLY the facts above. recommendations must be []."""
        parsed = _invoke_gemini_json(payload)
        if parsed and isinstance(parsed.get("reply"), str):
            recs = _sanitize_recommendations(parsed.get("recommendations", []), allowed)
            end = bool(parsed.get("end_of_conversation"))
            return parsed["reply"], recs, end
        return (
            "The comparison service is temporarily unavailable. Please try again shortly.",
            [],
            False,
        )

    force_final = turns >= MAX_TURNS
    vague_first = turns == 1 and _is_vague_first_turn(last_user)

    if vague_first and not force_final:
        return (
            "To recommend the right SHL Individual Test assessments, what role are you hiring for "
            "and what skills or qualities do you want to measure (for example cognitive ability, coding, or personality)?",
            [],
            False,
        )

    tail = _conversation_tail(msgs)
    search_query = tail
    search_block = _search_context_block(search_query, top_k=12)

    turn_note = (
        f"USER_TURN_COUNT={turns}. "
        f"{'This is the final turn: you must return up to 10 recommendations from search results and close helpfully.' if force_final else ''}"
    )

    payload = f"""{turn_note}

CONVERSATION:
{tail}

{search_block}

Return JSON with reply, recommendations (only from search results; empty if you still need one clarifying question — except on final turn you must recommend from results), end_of_conversation (true if user seems satisfied after a shortlist)."""

    if force_final:
        payload += (
            "\nFinal-turn rule: recommendations must be non-empty unless search results are empty."
        )

    parsed = _invoke_gemini_json(payload)
    if not parsed or not isinstance(parsed.get("reply"), str):
        return (
            "I am having trouble reaching the recommendation engine right now. "
            "Please try again in a moment.",
            [],
            False,
        )

    recs = _sanitize_recommendations(parsed.get("recommendations", []), allowed)

    end = bool(parsed.get("end_of_conversation"))
    if force_final and recs:
        end = True
    return parsed["reply"], recs, end
