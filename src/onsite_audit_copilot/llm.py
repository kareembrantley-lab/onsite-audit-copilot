"""
Thin wrapper around the Claude API (text + vision), used for five tasks in
this pipeline: entity extraction, checklist tagging, checklist
recommendation, photo captioning/matching, and the executive-summary
synthesis. Every one of those has a rules-based fallback (see
entity_extraction.py, checklist.py, vision.py, findings.py), so the whole
pipeline still runs -- with blunter but still correct output -- when
ANTHROPIC_API_KEY isn't set. A reviewer cloning this repo should not need
Kareem's API key to see it work end to end.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL = os.environ.get("ONSITE_AUDIT_MODEL", "claude-sonnet-4-5")


@lru_cache(maxsize=1)
def _client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(api_key=api_key)


def llm_available() -> bool:
    return _client() is not None


def _call(prompt: str, max_tokens: int = 400, image_path: str | Path | None = None) -> str | None:
    client = _client()
    if client is None:
        return None
    content: list[dict] = []
    if image_path is not None:
        img_bytes = Path(image_path).read_bytes()
        media_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(img_bytes).decode()},
        })
    content.append({"type": "text", "text": prompt})
    try:
        response = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
    except Exception:  # noqa: BLE001 - never let an LLM hiccup break the pipeline
        return None


def extract_entities_llm(text: str) -> dict | None:
    """Returns {"equipment": [...], "process_step": str|None,
    "safety_issue": bool, "bottleneck_observation": bool} or None."""
    prompt = (
        "You are helping process field notes dictated by an operations auditor walking a "
        "facility floor. From the note below, extract: (1) any specific equipment or "
        "fixtures mentioned, (2) the single process step this note is about (e.g. "
        "'receiving', 'packing', 'assembly'), if identifiable, (3) whether this describes a "
        "safety issue, (4) whether this describes a bottleneck/backlog/queue condition. "
        "Reply with strict JSON only: {\"equipment\": [str], \"process_step\": str|null, "
        "\"safety_issue\": bool, \"bottleneck_observation\": bool}. No prose outside the JSON.\n\n"
        f"Note: {text}"
    )
    text_out = _call(prompt, max_tokens=250)
    if not text_out:
        return None
    try:
        return json.loads(text_out[text_out.index("{"): text_out.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def tag_note_llm(text: str, checklist_items: list[dict]) -> dict | None:
    """checklist_items: [{"id": str, "category": str, "prompt": str}, ...].
    Returns {"item_id": str|None, "rationale": str} or None."""
    prompt = (
        "You are tagging a field note from an operations audit against a fixed checklist. "
        "Pick the single best-matching checklist item id, or null if none genuinely apply -- "
        "do not force a match. Reply with strict JSON only: "
        "{\"item_id\": str|null, \"rationale\": \"<one sentence>\"}.\n\n"
        f"Checklist items:\n{json.dumps(checklist_items, indent=2)}\n\nField note: {text}"
    )
    text_out = _call(prompt, max_tokens=200)
    if not text_out:
        return None
    try:
        return json.loads(text_out[text_out.index("{"): text_out.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def recommend_checklist_llm(engagement_description: str, checklists: list[dict]) -> dict | None:
    """checklists: [{"key": str, "name": str, "description": str}, ...].
    Returns {"key": str, "rationale": str} or None."""
    prompt = (
        "An operations auditor described their upcoming site visit in one line. Pick which "
        "of the available audit checklists fits best. Reply with strict JSON only: "
        "{\"key\": str, \"rationale\": \"<one sentence>\"}.\n\n"
        f"Available checklists:\n{json.dumps(checklists, indent=2)}\n\n"
        f"Engagement description: {engagement_description}"
    )
    text_out = _call(prompt, max_tokens=150)
    if not text_out:
        return None
    try:
        return json.loads(text_out[text_out.index("{"): text_out.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def caption_photo_llm(image_path: str | Path, checklist_items: list[dict]) -> dict | None:
    """Returns {"caption": str, "item_id": str|null} or None."""
    prompt = (
        "You are captioning a photo taken during an operations audit floor walk, for a "
        "findings report. Write one factual sentence describing what the photo shows "
        "(no speculation), then pick the single best-matching checklist item id, or null "
        "if none genuinely apply. Reply with strict JSON only: "
        "{\"caption\": \"<one sentence>\", \"item_id\": str|null}.\n\n"
        f"Checklist items:\n{json.dumps(checklist_items, indent=2)}"
    )
    text_out = _call(prompt, max_tokens=200, image_path=image_path)
    if not text_out:
        return None
    try:
        return json.loads(text_out[text_out.index("{"): text_out.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def synthesize_executive_summary_llm(
    facility_name: str, visit_area: str, checklist_name: str, findings_summary: dict
) -> str | None:
    prompt = (
        "You are an operations consultant writing the executive summary of a one-page "
        "site-visit findings report. Write 3-4 sentences: state the overall read, name the "
        "one or two most important findings driving it, and note the single highest-priority "
        "action. Be direct and specific with counts/severities. No consultant jargon, no "
        "em-dashes.\n\n"
        f"Facility: {facility_name}\nArea: {visit_area}\nChecklist used: {checklist_name}\n"
        f"Findings summary: {json.dumps(findings_summary, default=str)}"
    )
    return _call(prompt, max_tokens=300)
