import re

_CHAPTER_LINE_RE = re.compile(
    r"^\s*\d+\.\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?)\s*$"
)


def _to_seconds(timestamp: str) -> int:
    parts = [int(p) for p in timestamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def parse_chapters(chapters_text: str) -> list:
    """Parse extract_action_items()'s '1. [MM:SS] Title\\nDescription' output
    into [{"time_seconds": int, "title": str, "description": str}, ...].
    Unparseable lines are skipped rather than raising."""
    if not chapters_text:
        return []

    chapters = []
    current = None

    for line in chapters_text.splitlines():
        match = _CHAPTER_LINE_RE.match(line)
        if match:
            if current:
                current["description"] = current["description"].strip()
                chapters.append(current)
            try:
                seconds = _to_seconds(match.group(1))
            except ValueError:
                current = None
                continue
            current = {
                "time_seconds": seconds,
                "title": match.group(2).strip(),
                "description": "",
            }
        elif current is not None and line.strip():
            current["description"] += line.strip() + " "

    if current:
        current["description"] = current["description"].strip()
        chapters.append(current)

    return chapters
