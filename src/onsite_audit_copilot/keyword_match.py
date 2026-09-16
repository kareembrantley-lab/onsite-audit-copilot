"""
Shared keyword-matching helper with a lightweight negation guard, used by
both entity_extraction.py and checklist.py's rules-based fallback.

A naive `keyword in text` check can't tell "a backlog at the shrink-wrap
station" from "no backlog there, everything's running fine" -- both contain
the literal substring "backlog". This guard looks a short window backward
from each match for a negation cue and drops that match if one is found,
so a note that explicitly states a condition is ABSENT doesn't get tagged
as though it were present. It uses no real NLP and is not sound over
arbitrary sentence structure -- "not unlike a bottleneck" would still slip
through -- but it correctly handles the common, cleanly worded case a
field auditor's dictation actually produces. See docs/architecture.md for
the honestly-disclosed limitation; the LLM-assisted path in llm.py doesn't
have this gap at all, since it reads the whole sentence rather than
scanning for substrings.
"""

from __future__ import annotations

_NEGATION_CUES = [
    "no ", "not ", "isn't ", "wasn't ", "weren't ", "aren't ", "hasn't ", "haven't ",
    "doesn't ", "didn't ", "won't ", "without ", "never ", "no longer ",
]
_WINDOW = 28


def find_matches(text_lower: str, vocab: list[str]) -> list[str]:
    """Returns the subset of `vocab` whose terms appear in `text_lower` and
    are not immediately preceded by a negation cue."""
    hits = []
    for term in vocab:
        idx = text_lower.find(term)
        if idx == -1:
            continue
        window = text_lower[max(0, idx - _WINDOW):idx]
        if any(cue in window for cue in _NEGATION_CUES):
            continue
        hits.append(term)
    return hits


def best_match(text_lower: str, candidates: list[tuple[str, list[str]]]) -> tuple[str | None, list[str]]:
    """candidates: [(key, keywords), ...]. Returns (best_key, best_hits),
    scoring each candidate by (hit count, length of its longest matched
    keyword) so a more specific match (e.g. "waiting on a part") beats a
    shorter one that happens to be a substring of it ("waiting on") even
    when both register exactly one hit -- a plain hit-count tie would
    otherwise always resolve to whichever candidate is checked first,
    regardless of which phrase is actually the better match."""
    best_key, best_hits, best_score = None, [], (0, 0)
    for key, keywords in candidates:
        hits = find_matches(text_lower, keywords)
        if not hits:
            continue  # a candidate with zero hits must never win -- (0, 0) is the floor, not a valid score to beat
        score = (len(hits), max(len(h) for h in hits))
        if score > best_score:
            best_key, best_hits, best_score = key, hits, score
    return best_key, best_hits
