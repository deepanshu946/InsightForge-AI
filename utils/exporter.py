import re

from fpdf import FPDF
from fpdf.enums import XPos, YPos

_SECTIONS = [
    ("Executive Summary", "summary"),
    ("Chapters", "action_items"),
    ("Key Takeaways", "key_decisions"),
    ("Questions & Answers", "open_questions"),
]


def safe_filename(title: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", title or "report").strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:60] or "report"


def _latin1(text: str) -> str:
    """fpdf2's built-in core fonts only support Latin-1; degrade gracefully
    rather than crashing on smart quotes / emoji / non-Latin scripts."""
    return (text or "").encode("latin-1", "replace").decode("latin-1")


def build_txt_report(result: dict, include_transcript: bool = False) -> bytes:
    lines = [result.get("title", "Untitled Video"), "=" * 60, ""]

    for heading, key in _SECTIONS:
        lines.append(heading)
        lines.append("-" * len(heading))
        lines.append(result.get(key, "") or "")
        lines.append("")

    if include_transcript:
        lines.append("Transcript")
        lines.append("-" * 10)
        lines.append(result.get("transcript", "") or "")

    return "\n".join(lines).encode("utf-8")


def build_pdf_report(result: dict, include_transcript: bool = False) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, _latin1(result.get("title", "Untitled Video")), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    sections = list(_SECTIONS)
    if include_transcript:
        sections = sections + [("Transcript", "transcript")]

    for heading, key in sections:
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, _latin1(heading), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _latin1(result.get(key, "") or ""), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    output = pdf.output(dest="S")
    return bytes(output)
