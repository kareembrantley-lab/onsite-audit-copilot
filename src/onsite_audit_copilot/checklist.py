"""
Loads the swappable YAML checklist configs (data/checklists/*.yaml) and
does the two jobs that make the checklist "swappable" mean something in
practice: recommending which one fits a one-line engagement description
(the build spec's stretch goal), and tagging a field note against whichever
checklist is in use.

Both tasks are LLM-assisted with a keyword-rules fallback -- same pattern
as entity_extraction.py. Nothing about which checklist gets used, or how a
note gets tagged to it, requires ANTHROPIC_API_KEY to be set.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .keyword_match import best_match, find_matches
from .llm import llm_available, recommend_checklist_llm, tag_note_llm
from .schema import ChecklistConfig, ChecklistItem, ExtractedEntities, TaggedNote

_CHECKLIST_DIR = Path(__file__).resolve().parents[2] / "data" / "checklists"


def list_checklists(checklist_dir: Path | None = None) -> list[ChecklistConfig]:
    checklist_dir = checklist_dir or _CHECKLIST_DIR
    configs = []
    for path in sorted(checklist_dir.glob("*.yaml")):
        configs.append(_load_config(path))
    return configs


def load_checklist(key: str, checklist_dir: Path | None = None) -> ChecklistConfig:
    checklist_dir = checklist_dir or _CHECKLIST_DIR
    path = checklist_dir / f"{key}.yaml"
    if not path.exists():
        available = [p.stem for p in checklist_dir.glob("*.yaml")]
        raise ValueError(f"No checklist '{key}' -- available: {available}")
    return _load_config(path)


def _load_config(path: Path) -> ChecklistConfig:
    raw = yaml.safe_load(path.read_text())
    items = [
        ChecklistItem(
            id=i["id"], category=i["category"], prompt=i["prompt"],
            default_severity=i["default_severity"], keywords=i.get("keywords", []),
        )
        for i in raw["items"]
    ]
    return ChecklistConfig(
        key=raw["key"], name=raw["name"], description=raw["description"],
        recommend_when=raw.get("recommend_when", []), items=items,
    )


def recommend_checklist(engagement_description: str, checklist_dir: Path | None = None) -> tuple[ChecklistConfig, str, str]:
    """Returns (chosen_config, rationale, method)."""
    configs = list_checklists(checklist_dir)

    if llm_available():
        result = recommend_checklist_llm(
            engagement_description,
            [{"key": c.key, "name": c.name, "description": c.description} for c in configs],
        )
        if result and result.get("key"):
            match = next((c for c in configs if c.key == result["key"]), None)
            if match is not None:
                return match, result.get("rationale", ""), "llm"

    return _rules_recommend(engagement_description, configs)


def _rules_recommend(engagement_description: str, configs: list[ChecklistConfig]) -> tuple[ChecklistConfig, str, str]:
    lower = engagement_description.lower()
    scored = []
    for config in configs:
        hits = find_matches(lower, [kw.lower() for kw in config.recommend_when])
        scored.append((len(hits), config, hits))
    scored.sort(key=lambda t: -t[0])

    best_count, best_config, best_hits = scored[0]
    if best_count == 0:
        # No keyword signal at all -- default to the safety walk rather than
        # guessing at organization vs. throughput, since a missed safety
        # issue is the costliest kind of wrong guess here.
        fallback = next(c for c in configs if c.key == "safety_compliance")
        return fallback, "No keyword match in the engagement description; defaulted to the safety & compliance walk as the safer assumption.", "rules"

    rationale = f"Matched on: {', '.join(best_hits)}."
    return best_config, rationale, "rules"


def tag_note_to_checklist(text: str, entities: ExtractedEntities, checklist: ChecklistConfig, note_id: str) -> TaggedNote:
    if llm_available():
        result = tag_note_llm(
            text, [{"id": i.id, "category": i.category, "prompt": i.prompt} for i in checklist.items],
        )
        if result is not None:
            item_id = result.get("item_id")
            if item_id and checklist.get_item(item_id):
                return TaggedNote(
                    note_id=note_id, source_text=text, checklist_item_id=item_id,
                    confidence="high", rationale=result.get("rationale", ""),
                    entities=entities, method="llm",
                )
            return TaggedNote(
                note_id=note_id, source_text=text, checklist_item_id=None,
                confidence="low", rationale="LLM found no confident match against this checklist.",
                entities=entities, method="llm",
            )

    return _rules_tag(text, entities, checklist, note_id)


def _rules_tag(text: str, entities: ExtractedEntities, checklist: ChecklistConfig, note_id: str) -> TaggedNote:
    lower = text.lower()
    # best_match scores each item by (hit count, longest matched keyword),
    # not hit count alone -- a plain-count tie would otherwise always
    # resolve to whichever item happens to be listed first in the YAML,
    # even when a competing item matched a longer, more specific phrase
    # (e.g. "waiting on a part" should beat "waiting on").
    best_item_id, best_hits = best_match(lower, [(item.id, item.keywords) for item in checklist.items])

    if best_item_id is None:
        return TaggedNote(
            note_id=note_id, source_text=text, checklist_item_id=None,
            confidence="low", rationale="No checklist keyword matched this note.",
            entities=entities, method="unmatched",
        )

    confidence = "high" if len(best_hits) >= 2 else "medium"
    return TaggedNote(
        note_id=note_id, source_text=text, checklist_item_id=best_item_id,
        confidence=confidence, rationale=f"Matched keyword(s): {', '.join(best_hits)}.",
        entities=entities, method="rules",
    )
