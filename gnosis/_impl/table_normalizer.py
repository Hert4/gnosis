"""Normalize parsed Tables into LLM-friendly markdown.

Strategy:
  - Small tables (≤ KV_THRESHOLD cols) → markdown pipe table (compact)
  - Wide tables (> KV_THRESHOLD cols) → KV render per row
    (each row becomes "**Dòng N:**\n  - col_name: value" blocks)
  - Skip template rows (>= TEMPLATE_EMPTY_RATIO empty cells)
  - Skip column-reference rows (short alphanumeric placeholders like "A, B, 1, 2, 3")
  - Flatten multi-row headers into compound "Parent > Child" via Table.flat_headers()
"""

from __future__ import annotations

import re

from gnosis._impl.html_table_parser import find_table_blocks, parse_single_table
from gnosis._impl.native_schema import Table

# Threshold to switch from pipe table to KV rendering
KV_THRESHOLD = 6

# Row is "template" if this share of its cells are empty
TEMPLATE_EMPTY_RATIO = 0.7

# Short placeholder chars that signal a column-reference row (A, B, C, 1, 2, ...)
_PLACEHOLDER_RE = re.compile(r"^[A-Za-z]$|^\d{1,2}$")


def is_template_row(cells: list[str]) -> bool:
    """True if >= TEMPLATE_EMPTY_RATIO of cells are empty or trivial."""
    if not cells:
        return True
    empty = sum(
        1 for c in cells
        if not c.strip() or c.strip() in {".", "-", "_", "/", ""}
    )
    return empty / len(cells) >= TEMPLATE_EMPTY_RATIO


def is_column_ref_row(cells: list[str]) -> bool:
    """True if row looks like '(A) (B) (C) 1 2 3 ...' — column references, not data."""
    if not cells:
        return False
    non_empty = [c.strip() for c in cells if c.strip()]
    if len(non_empty) < 3:
        return False
    placeholders = sum(1 for c in non_empty if _PLACEHOLDER_RE.match(c))
    return placeholders / len(non_empty) >= 0.8


def _render_markdown_pipe(table: Table) -> str:
    """Render narrow table as markdown pipe format."""
    headers = table.flat_headers()
    body = table.body_rows()

    # Filter rows
    body = [r for r in body if not is_template_row(r) and not is_column_ref_row(r)]

    if not headers and not body:
        return ""

    lines: list[str] = []
    header_line = "| " + " | ".join(h or " " for h in headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines.append(header_line)
    lines.append(sep_line)
    for row in body:
        # Pad row to header length
        padded = row + [""] * (len(headers) - len(row))
        padded = padded[:len(headers)]
        lines.append("| " + " | ".join(c or " " for c in padded) + " |")
    return "\n".join(lines)


def _render_kv(table: Table) -> str:
    """Render wide table as KV blocks: one block per body row.

    For each data row, emit:
        **Dòng N:**
          - <flat_header>: <value>
          - ...
    Cells with empty values are skipped to reduce noise.
    """
    headers = table.flat_headers()
    body = table.body_rows()

    # Filter rows
    body = [r for r in body if not is_template_row(r) and not is_column_ref_row(r)]

    if not body:
        # Still return header schema so LLM knows table structure exists
        header_list = "\n".join(f"  - {h}" for h in headers if h)
        return (
            f"<!-- TABLE: wide table ({table.n_cols} cols, empty/template) -->\n"
            f"**Schema:**\n{header_list}\n"
        )

    out: list[str] = []
    out.append(f"<!-- TABLE: wide table ({table.n_cols} cols, {len(body)} data rows) -->")
    # Emit schema header for LLM context
    schema_line = " | ".join(h for h in headers if h)
    out.append(f"**Schema:** {schema_line}\n")

    for i, row in enumerate(body, 1):
        out.append(f"**Dòng {i}:**")
        for h, v in zip(headers, row):
            v_stripped = v.strip()
            if v_stripped:
                out.append(f"  - {h or f'col_{headers.index(h)+1}'}: {v_stripped}")
        out.append("")
    return "\n".join(out)


def render_table(table: Table) -> str:
    """Dispatch to pipe or KV rendering based on column count."""
    if table.n_cols == 0:
        return ""
    if table.n_cols <= KV_THRESHOLD:
        return _render_markdown_pipe(table)
    return _render_kv(table)


def normalize_tables_in_markdown(markdown: str, page: int) -> str:
    """Replace every <table>...</table> block with normalized markdown.

    Preserves surrounding text (headings, paragraphs) as-is. Only the HTML
    table blocks get rewritten.
    """
    blocks = find_table_blocks(markdown)
    if not blocks:
        return markdown

    out_parts: list[str] = []
    cursor = 0
    for start, end, html in blocks:
        # Text before this table
        out_parts.append(markdown[cursor:start])
        # Normalize this table
        table = parse_single_table(html, page=page)
        rendered = render_table(table)
        if rendered:
            out_parts.append(rendered)
        else:
            # Fallback: keep original HTML if parsing failed
            out_parts.append(html)
        cursor = end
    # Tail
    out_parts.append(markdown[cursor:])

    return "".join(out_parts)
