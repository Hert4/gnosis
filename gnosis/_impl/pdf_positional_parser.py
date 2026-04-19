"""
pdf_positional_parser.py — Convert PDF → HTML files using pixel coordinates.

Uses pdftohtml -xml to get exact coordinates of each text element,
then reconstructs layout into structured HTML.

General for all document types:
  - Plain text → <p>, <h2>/<h3>
  - Tables (multiple even columns) → <table> with cells
  - Mixed → auto-detect per page
"""

import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# XML parsing
# ─────────────────────────────────────────────────────────────

def _run_pdftohtml(pdf_path: str) -> ET.Element:
    """Run pdftohtml -xml and return parsed XML root.

    On Windows, pdftohtml cannot handle non-ASCII paths.
    If the path contains non-ASCII chars, copy to a temp file first.
    """
    path = str(pdf_path)
    tmp_copy = None

    try:
        # Check if path is ASCII-safe for pdftohtml
        path.encode("ascii")
    except UnicodeEncodeError:
        # Copy to temp file with ASCII-safe name
        suffix = Path(path).suffix
        fd, tmp_copy = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        shutil.copy2(path, tmp_copy)
        path = tmp_copy

    try:
        xml_bytes = subprocess.check_output(
            ["pdftohtml", "-xml", "-stdout", "-enc", "UTF-8", path],
            stderr=subprocess.DEVNULL,
        )
    finally:
        if tmp_copy and os.path.exists(tmp_copy):
            os.unlink(tmp_copy)

    xml_str = xml_bytes.decode("utf-8", errors="replace")
    xml_str = re.sub(r"<!DOCTYPE[^\[]*\[[^\]]*\]>", "", xml_str)
    xml_str = re.sub(r"<!DOCTYPE[^>]*>", "", xml_str)
    return ET.fromstring(xml_str)


def _extract_items(page_elem) -> list:
    """
    Extract all text items from page element.
    Returns: list of (top, left, width, height, text, is_bold)
    """
    items = []
    for t in page_elem.findall("text"):
        top    = int(t.get("top", 0))
        left   = int(t.get("left", 0))
        width  = int(t.get("width", 0))
        height = int(t.get("height", 12))
        text   = "".join(t.itertext()).strip()
        is_bold = t.find("b") is not None
        if text:
            items.append((top, left, width, height, text, is_bold))
    return items


# ─────────────────────────────────────────────────────────────
# Layout detection
# ─────────────────────────────────────────────────────────────

def _group_rows(items: list, tolerance: int = 6) -> dict:
    """
    Group items into rows by top coordinate (±tolerance px).
    Returns: {canonical_top: [(left, width, height, text, is_bold)]}
    """
    rows = {}
    for top, left, width, height, text, is_bold in sorted(items, key=lambda x: x[0]):
        matched = next((k for k in rows if abs(k - top) <= tolerance), None)
        if matched is None:
            matched = top
        rows.setdefault(matched, []).append((left, width, height, text, is_bold))
    return rows


def _detect_columns(rows: dict, page_width: int) -> list:
    """
    Detect column boundaries based on frequency of left positions.
    Cluster nearby left values (≤30px) → canonical column.
    Returns: sorted list of canonical left positions (column starts).
    """
    left_freq = Counter()
    for items in rows.values():
        for left, *_ in items:
            left_freq[left] += 1

    sorted_lefts = sorted(left_freq.keys())
    if not sorted_lefts:
        return []

    clusters = [[sorted_lefts[0]]]
    for l in sorted_lefts[1:]:
        if l - clusters[-1][-1] <= 30:
            clusters[-1].append(l)
        else:
            clusters.append([l])

    col_starts = []
    for cl in clusters:
        total = sum(left_freq[l] for l in cl)
        canon = int(sum(l * left_freq[l] for l in cl) / total)
        col_starts.append(canon)

    return col_starts


def _assign_col(left: int, col_starts: list) -> int:
    """Assign nearest column index."""
    return min(range(len(col_starts)), key=lambda i: abs(left - col_starts[i]))


def _is_table_page(rows: dict, col_starts: list) -> bool:
    """
    Heuristic: page has table if:
    - ≥4 columns detected
    - >40% of rows have ≥3 populated cells
    """
    if len(col_starts) < 4:
        return False

    populated_rows = 0
    for items in rows.values():
        used_cols = set(_assign_col(left, col_starts) for left, *_ in items)
        if len(used_cols) >= 3:
            populated_rows += 1

    return populated_rows > len(rows) * 0.4


# ─────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────

def _rows_to_html_table(rows: dict, col_starts: list) -> str:
    """Convert rows → HTML table with multi-line cell merging."""
    sorted_tops = sorted(rows.keys())

    raw_matrix = []
    for top in sorted_tops:
        cells = defaultdict(list)
        for left, width, height, text, is_bold in sorted(rows[top]):
            c = _assign_col(left, col_starts)
            cells[c].append(("<b>" + text + "</b>") if is_bold else text)
        raw_matrix.append(dict(cells))

    # Merge continuation rows
    merged = []
    i = 0
    while i < len(raw_matrix):
        current = {c: list(parts) for c, parts in raw_matrix[i].items()}
        while i + 1 < len(raw_matrix):
            nxt = raw_matrix[i + 1]
            if 0 not in nxt and len(nxt) <= max(2, len(col_starts) // 2):
                for c, parts in nxt.items():
                    current.setdefault(c, []).extend(parts)
                i += 1
            else:
                break
        merged.append(current)
        i += 1

    lines = ["<table>"]
    for row_dict in merged:
        lines.append("  <tr>")
        for c in range(len(col_starts)):
            parts = row_dict.get(c, [])
            cell_text = " ".join(parts)
            tag = "th" if any("<b>" in p for p in parts) and c == 0 else "td"
            lines.append(f"    <{tag}>{cell_text}</{tag}>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _rows_to_html_text(rows: dict, page_width: int) -> str:
    """Convert rows → HTML paragraphs/headings."""
    sorted_tops = sorted(rows.keys())

    all_heights = [h for items in rows.values() for _, _, h, _, _ in items]
    median_h = sorted(all_heights)[len(all_heights) // 2] if all_heights else 12

    lines_html = []
    for top in sorted_tops:
        items_in_row = sorted(rows[top])
        parts = []
        for left, width, height, text, is_bold in items_in_row:
            if is_bold:
                parts.append(f"<strong>{text}</strong>")
            else:
                parts.append(text)
        line_text = " ".join(parts)

        first_height = items_in_row[0][2]
        first_bold = items_in_row[0][4]
        if first_bold and first_height > median_h:
            tag = "h2"
        elif first_bold:
            tag = "h3"
        else:
            tag = "p"

        lines_html.append(f"<{tag}>{line_text}</{tag}>")

    return "\n".join(lines_html)


def _page_to_html(page_elem, page_num: int) -> str:
    """Convert 1 page element → full HTML string."""
    page_width  = int(page_elem.get("width", 800))
    page_height = int(page_elem.get("height", 1000))

    items = _extract_items(page_elem)
    if not items:
        return ""

    rows = _group_rows(items)
    col_starts = _detect_columns(rows, page_width)

    if _is_table_page(rows, col_starts):
        body = _rows_to_html_table(rows, col_starts)
        page_type = "table"
    else:
        body = _rows_to_html_text(rows, page_width)
        page_type = "text"

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="page" content="{page_num}">
  <meta name="type" content="{page_type}">
  <title>Page {page_num}</title>
</head>
<body>
<section id="page-{page_num}" data-page="{page_num}" data-type="{page_type}">
{body}
</section>
</body>
</html>"""
