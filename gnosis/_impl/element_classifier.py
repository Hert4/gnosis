"""Classify text elements (captions, footnotes) in OCR2 markdown.

Uses regex pattern matching (no ML). Wraps detected elements with HTML
comment markers that downstream consumers can use to filter/emphasize.

Patterns are tuned for Vietnamese legal/accounting documents but generic.
"""

from __future__ import annotations

import re

# Caption patterns — "Hình 1:", "Bảng 2.3 -", "Biểu X:", "Figure 1.1:", "Table 2.3.4 —"
# Matches start of line, allows period/dash/colon as separator.
CAPTION_RE = re.compile(
    r"^(?P<kind>Hình|Bảng|Biểu|Sơ đồ|Figure|Table|Chart|Diagram)"
    r"\s+(?P<num>[\d]+(?:\.[\d]+)*)"
    r"\s*[:.\-–—]\s*(?P<rest>.+)$",
    re.MULTILINE | re.IGNORECASE,
)

# Footnote patterns — "(1) note", "(*) note", "[1] note" at start of line.
# Short (≤ 200 chars) so it doesn't eat whole paragraphs.
FOOTNOTE_RE = re.compile(
    r"^(?P<marker>\(\d{1,3}\)|\(\*+\)|\[\d{1,3}\]|\*{1,3})\s+(?P<text>.{5,200})$",
    re.MULTILINE,
)


def tag_captions(markdown: str) -> tuple[str, list[dict]]:
    """Wrap caption lines with HTML comment markers.

    Returns:
        (tagged_markdown, captions_metadata_list)
    """
    captions: list[dict] = []

    def _sub(m: re.Match) -> str:
        captions.append({
            "kind": m.group("kind"),
            "number": m.group("num"),
            "text": m.group("rest").strip(),
        })
        return (
            f"<!-- CAPTION: {m.group('kind')} {m.group('num')} -->\n"
            f"{m.group(0)}"
        )

    tagged = CAPTION_RE.sub(_sub, markdown)
    return tagged, captions


def tag_footnotes(markdown: str) -> tuple[str, list[dict]]:
    """Wrap footnote lines with HTML comment markers."""
    footnotes: list[dict] = []

    def _sub(m: re.Match) -> str:
        footnotes.append({
            "marker": m.group("marker"),
            "text": m.group("text").strip(),
        })
        return f"<!-- FOOTNOTE: {m.group('marker')} -->\n{m.group(0)}"

    tagged = FOOTNOTE_RE.sub(_sub, markdown)
    return tagged, footnotes


def tag_elements(markdown: str) -> tuple[str, dict]:
    """Run all classifiers, return (tagged_md, metadata dict).

    metadata keys: 'captions', 'footnotes'
    """
    md, captions = tag_captions(markdown)
    md, footnotes = tag_footnotes(md)
    return md, {"captions": captions, "footnotes": footnotes}
