"""
Speech-to-text for voice field notes. Live mode calls OpenAI's Whisper
transcription API (the industry-standard hosted STT endpoint -- this repo
treats it the same way places_signal.py in the audit-agent project treats
Google Places: an official, licensed API, not a workaround).

This demo deliberately does not synthesize fake audio files. Generating
speech audio realistic enough to be worth transcribing is its own project;
faking that step would test nothing real. Instead, the synthetic dataset
ships pre-transcribed dictation text -- the honest equivalent of "the
auditor's phone already transcribed this in the field" -- and
`get_transcript()` in fixture mode returns that text directly, with a
`method` field on every `TranscribedNote` disclosing exactly which path
produced it so a reviewer never mistakes one for the other.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .schema import TranscribedNote


@lru_cache(maxsize=1)
def _client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
    except ImportError:
        return None
    return openai.OpenAI(api_key=api_key)


def stt_available() -> bool:
    return _client() is not None


def _confidence_from_length(text: str) -> str:
    # A real Whisper response includes segment-level confidence; the demo
    # fixture path has no such signal, so this is a conservative proxy: a
    # very short transcript is more likely a mis-hearing or a dropped
    # recording than a fully captured observation.
    words = len(text.split())
    if words < 4:
        return "low"
    if words < 12:
        return "medium"
    return "high"


def get_transcript(note_id: str, audio_path: str | Path | None = None, fixture_transcript: str | None = None) -> TranscribedNote:
    if audio_path is not None:
        client = _client()
        if client is None:
            raise ValueError(
                "audio_path was given but OPENAI_API_KEY is not set -- "
                "pass fixture_transcript for demo mode instead, or set the key for live mode."
            )
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(model="whisper-1", file=f)
        text = result.text.strip()
        return TranscribedNote(note_id=note_id, text=text, confidence=_confidence_from_length(text), method="live_whisper")

    if fixture_transcript is not None:
        return TranscribedNote(
            note_id=note_id, text=fixture_transcript.strip(),
            confidence=_confidence_from_length(fixture_transcript), method="fixture",
        )

    raise ValueError("Provide audio_path (live mode) or fixture_transcript (demo mode).")
