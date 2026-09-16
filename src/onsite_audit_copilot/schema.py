"""
Core data contracts shared across every stage of the pipeline. Every stage
in pipeline.py consumes one of these and produces the next -- see
docs/architecture.md for the full data-flow diagram.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Severity scale, worst to best. Kept as an ordered list (not an IntEnum) so
# severity.py can do simple index arithmetic to "escalate by one level"
# without a separate mapping table to keep in sync.
SEVERITY_ORDER = ["critical", "high", "medium", "low", "observation"]


def escalate(severity: str, levels: int = 1) -> str:
    """Move a severity toward 'critical' by `levels` steps, clamped at the
    top of the scale. Escalating "critical" is a no-op, not an error --
    callers apply this speculatively without checking first."""
    idx = SEVERITY_ORDER.index(severity)
    return SEVERITY_ORDER[max(0, idx - levels)]


@dataclass
class FieldNoteInput:
    """One raw observation as captured on-site, before any processing."""
    note_id: str
    kind: str  # "voice" | "photo"
    timestamp: str  # ISO 8601
    # voice notes:
    audio_path: str | None = None          # live mode
    fixture_transcript: str | None = None  # demo mode -- see transcription.py
    # photo notes:
    image_path: str | None = None
    fixture_caption: dict | None = None    # sidecar metadata -- see vision.py


@dataclass
class TranscribedNote:
    note_id: str
    text: str
    confidence: str  # "low" | "medium" | "high"
    method: str      # "live_whisper" | "fixture"


@dataclass
class ExtractedEntities:
    equipment: list[str] = field(default_factory=list)
    process_step: str | None = None
    safety_issue: bool = False
    safety_keywords_matched: list[str] = field(default_factory=list)
    bottleneck_observation: bool = False
    bottleneck_keywords_matched: list[str] = field(default_factory=list)
    method: str = "rules"  # "llm" | "rules"


@dataclass
class ChecklistItem:
    id: str
    category: str
    prompt: str
    default_severity: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class ChecklistConfig:
    key: str
    name: str
    description: str
    recommend_when: list[str]
    items: list[ChecklistItem]

    def get_item(self, item_id: str) -> ChecklistItem | None:
        return next((i for i in self.items if i.id == item_id), None)


@dataclass
class TaggedNote:
    note_id: str
    source_text: str
    checklist_item_id: str | None  # None if nothing matched confidently
    confidence: str  # "low" | "medium" | "high"
    rationale: str
    entities: ExtractedEntities
    method: str  # "llm" | "rules" | "unmatched"


@dataclass
class PhotoMatch:
    note_id: str
    image_path: str
    caption: str
    checklist_item_id: str | None
    confidence: str
    method: str  # "llm_vision" | "fixture_metadata"


@dataclass
class EvidenceItem:
    """One traceable piece of evidence behind a Finding -- kept as its own
    note_id so the report's appendix can point back to the exact source
    note or photo, not just an anonymous excerpt."""
    note_id: str
    kind: str  # "text" | "photo"
    detail: str  # note text, or photo caption
    image_filename: str | None = None


@dataclass
class Finding:
    checklist_item_id: str
    category: str
    title: str
    severity: str
    occurrence_count: int
    evidence: list[EvidenceItem] = field(default_factory=list)
    narrative: str = ""
    escalation_reasons: list[str] = field(default_factory=list)

    @property
    def photo_refs(self) -> list[str]:
        return [e.image_filename for e in self.evidence if e.image_filename]


@dataclass
class ActionItem:
    priority_rank: int
    finding_title: str
    severity: str
    recommended_action: str


@dataclass
class FindingsReport:
    facility_name: str
    visit_area: str
    visit_date: str
    checklist: ChecklistConfig
    executive_summary: str
    executive_summary_method: str  # "llm" | "template"
    findings: list[Finding]
    action_list: list[ActionItem]
    notes_processed: int
    photos_processed: int
    unmatched_notes: int
    synthetic_data_note: str | None = None
