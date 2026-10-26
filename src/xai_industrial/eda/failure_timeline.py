from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FailureEvent:
    event_id: str
    start_time: str
    end_time: str
    event_type: str
    source: str
    verification_status: str
    confidence: str
    notes: str = ""


FAILURE_TIMELINE_NOT_AVAILABLE = "Not yet available"


def empty_failure_timeline() -> list[FailureEvent]:
    return []


def failure_timeline_status(events: list[FailureEvent]) -> str:
    if not events:
        return FAILURE_TIMELINE_NOT_AVAILABLE
    return "Verified event records available"


def write_failure_timeline_csv(events: list[FailureEvent], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "start_time",
        "end_time",
        "event_type",
        "source",
        "verification_status",
        "confidence",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))
