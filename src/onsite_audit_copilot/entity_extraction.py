"""
Turns a transcribed field note into structured entities: what equipment it
mentions, what process step it's about, and whether it reads as a safety
issue or a bottleneck observation. LLM-assisted if ANTHROPIC_API_KEY is set
(llm.extract_entities_llm), keyword rules otherwise -- same fallback shape
as every other LLM touchpoint in this pipeline.

The rules-based path matters more here than it does in a review-sentiment
tool: a safety/bottleneck miss doesn't just weaken a score, it can mean a
real finding never reaches the report. So the keyword lists below were
built to be recall-first (catch the observation, even loosely worded) --
checklist.py's tagging step, not this one, is where precision matters.
"""

from __future__ import annotations

from .keyword_match import find_matches
from .llm import extract_entities_llm, llm_available
from .schema import ExtractedEntities

EQUIPMENT_VOCAB = [
    "conveyor", "forklift", "pallet jack", "shrink-wrap machine", "shrink wrapper",
    "label printer", "scanner", "workstation", "assembly line", "torque wrench",
    "cart", "rack", "scale", "dock door", "dock plate", "man-lift", "ladder",
    "extension cord", "fire extinguisher", "eyewash station", "guard rail",
    "machine guard", "hand truck", "tote", "bin",
]

PROCESS_STEP_VOCAB = [
    "receiving", "staging", "put-away", "picking", "packing", "shrink-wrap",
    "labeling", "quality check", "qc inspection", "assembly", "sub-assembly",
    "changeover", "shipping", "loading", "kitting",
]

SAFETY_KEYWORDS = [
    "unguarded", "no guard", "missing guard", "guard removed", "guard bypassed",
    "exposed pinch point", "pinch point", "blocked exit", "blocked egress",
    "obstructed exit", "extinguisher expired", "no ppe", "without ppe",
    "no safety glasses", "no hearing protection", "no hard hat", "no gloves",
    "trip hazard", "slip hazard", "spill", "frayed cord", "exposed wiring",
    "near miss", "unlabeled chemical", "not wearing",
]

BOTTLENECK_KEYWORDS = [
    "queue", "backlog", "piled up", "piling up", "waiting on", "waiting for",
    "behind schedule", "idle", "starved", "blocked by", "bottleneck",
    "wip building up", "build-up", "overflowing", "running ahead",
    "overproducing",
]


def _rules_extract(text: str) -> ExtractedEntities:
    lower = text.lower()
    equipment = find_matches(lower, EQUIPMENT_VOCAB)
    process_step = next((step for step in PROCESS_STEP_VOCAB if step in lower), None)
    safety_matches = find_matches(lower, SAFETY_KEYWORDS)
    bottleneck_matches = find_matches(lower, BOTTLENECK_KEYWORDS)

    return ExtractedEntities(
        equipment=equipment,
        process_step=process_step,
        safety_issue=bool(safety_matches),
        safety_keywords_matched=safety_matches,
        bottleneck_observation=bool(bottleneck_matches),
        bottleneck_keywords_matched=bottleneck_matches,
        method="rules",
    )


def extract_entities(text: str) -> ExtractedEntities:
    if llm_available():
        result = extract_entities_llm(text)
        if result is not None:
            # Even in LLM mode, keep the keyword evidence -- it's what the
            # report cites in "why this was flagged safety/bottleneck," and
            # an LLM call doesn't return which words tripped it.
            lower = text.lower()
            return ExtractedEntities(
                equipment=result.get("equipment") or [],
                process_step=result.get("process_step"),
                safety_issue=bool(result.get("safety_issue")),
                safety_keywords_matched=find_matches(lower, SAFETY_KEYWORDS),
                bottleneck_observation=bool(result.get("bottleneck_observation")),
                bottleneck_keywords_matched=find_matches(lower, BOTTLENECK_KEYWORDS),
                method="llm",
            )
    return _rules_extract(text)
