"""Resuming a build: what a finished stage of the work directory records.

A stage that finishes writes `<stage>.done` beside its output, naming the
inputs it was built from. A later run skips the stage when the record
matches, rebuilds it when there is none (it never finished, so its output is
partial), and stops when the record names other inputs: the work directory
belongs to another release, and reusing it would mix the two.
"""

from __future__ import annotations

import json
from pathlib import Path


def is_done(output: Path, inputs: dict) -> bool:
    """Whether `output` finished, built from `inputs`."""
    marker = output.with_name(output.name + ".done")
    if not marker.exists():
        return False
    recorded = json.loads(marker.read_text())
    if recorded != inputs:
        differ = ", ".join(f"{key} {recorded.get(key)!r} there, {inputs.get(key)!r} here"
                           for key in sorted(set(recorded) | set(inputs))
                           if recorded.get(key) != inputs.get(key))
        raise SystemExit(f"{output} was built from other inputs ({differ}); "
                         "give this build its own work directory")
    return True


def mark_done(output: Path, inputs: dict) -> None:
    """Record that `output` finished, built from `inputs`."""
    marker = output.with_name(output.name + ".done")
    marker.write_text(json.dumps(inputs, indent=2, sort_keys=True))
