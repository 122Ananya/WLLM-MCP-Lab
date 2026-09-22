"""MCP server for a personal-assistant notes store.

Exposes two tools backed by a local notes.json file:
  - save_note(content, tags): append a new note
  - search_notes(query): return notes whose content or tags match the query
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

NOTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes.json")

mcp = FastMCP("personal-assistant-notes")


def _load_notes() -> List[dict]:
    if not os.path.exists(NOTES_PATH):
        return []
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_notes(notes: List[dict]) -> None:
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)


@mcp.tool()
def save_note(content: str, tags: Optional[List[str]] = None) -> dict:
    """Save a new note.

    Args:
        content: The text of the note to remember.
        tags: Optional list of tags to categorize the note (e.g. ["work", "deadline"]).
    """
    notes = _load_notes()
    note = {
        "id": (notes[-1]["id"] + 1) if notes else 1,
        "content": content,
        "tags": tags or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    notes.append(note)
    _save_notes(notes)
    return {"status": "saved", "note": note}


@mcp.tool()
def search_notes(query: str) -> dict:
    """Search saved notes by keyword or tag.

    Args:
        query: Text to look for in note content or tags (case-insensitive).
    """
    notes = _load_notes()
    q = query.strip().lower()
    if not q:
        matches = notes
    else:
        matches = [
            n
            for n in notes
            if q in n["content"].lower() or any(q in t.lower() for t in n.get("tags", []))
        ]
    return {"count": len(matches), "results": matches}


if __name__ == "__main__":
    mcp.run()
