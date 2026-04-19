"""Parse HTML tables from OCR2 markdown output into Table dataclass.

Uses stdlib html.parser. Resolves colspan/rowspan into an explicit grid so
downstream normalization can skip empty cells, detect template rows, and
flatten multi-row headers into compound keys.

The OCR2 output interleaves `<table>...</table>` blocks within markdown text.
This module extracts each table block and parses its structure.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from gnosis._impl.native_schema import Cell, Table

_TABLE_BLOCK_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
_TABLE_OPEN_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)


def find_table_blocks(markdown: str) -> list[tuple[int, int, str]]:
    """Return list of (start_idx, end_idx, html) for each <table>...</table> block.

    Handles truncated tables (open <table> without closing </table>) by
    treating the rest of the markdown as the table body and auto-closing it.
    This is common when OCR2 hits max_tokens mid-table.
    """
    # Pass 1: complete tables
    complete: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group()) for m in _TABLE_BLOCK_RE.finditer(markdown)
    ]

    # Pass 2: detect truncated tables — <table> without matching </table>
    # Find all <table> openings NOT already covered by a complete block
    covered_ranges = [(s, e) for s, e, _ in complete]

    def _is_covered(pos: int) -> bool:
        return any(s <= pos < e for s, e in covered_ranges)

    truncated: list[tuple[int, int, str]] = []
    for m in _TABLE_OPEN_RE.finditer(markdown):
        if _is_covered(m.start()):
            continue
        # Take from <table> to end of markdown, auto-close
        start = m.start()
        end = len(markdown)
        body = markdown[start:end]
        auto_closed = body.rstrip() + "\n</table>"
        truncated.append((start, end, auto_closed))

    # Merge and sort
    all_blocks = complete + truncated
    all_blocks.sort(key=lambda x: x[0])
    return all_blocks


class _TableHTMLParser(HTMLParser):
    """Stateful parser that expands colspan/rowspan into a grid.

    Produces:
        header_rows: list[list[str]]  — each row has n_cols entries after expansion
        body_cells:  list[Cell]       — one Cell per logical grid position
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # Output
        self.header_rows: list[list[str]] = []
        self.body_cells: list[Cell] = []
        self.n_cols: int = 0

        # Grid expansion state — track active rowspans that occupy future rows
        # _pending_rowspans[col] = (remaining, text)
        self._pending_rowspans: dict[int, tuple[int, str]] = {}

        # Section state
        self._in_thead: bool = False
        self._in_tbody: bool = False
        self._in_row: bool = False
        self._current_row_cells: list[tuple[int, int, int, str, bool]] = []  # (col, rowspan, colspan, text, is_header)
        self._current_row_is_header: bool = False

        # Cell state
        self._in_cell: bool = False
        self._cell_is_header: bool = False
        self._cell_rowspan: int = 1
        self._cell_colspan: int = 1
        self._cell_buffer: list[str] = []

        # Row counter (for body cells only)
        self._body_row_idx: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "thead":
            self._in_thead = True
            self._in_tbody = False
        elif t == "tbody":
            self._in_tbody = True
            self._in_thead = False
        elif t == "tr":
            self._in_row = True
            self._current_row_cells = []
            self._current_row_is_header = False
        elif t in ("td", "th"):
            self._in_cell = True
            self._cell_is_header = (t == "th") or self._in_thead
            if self._cell_is_header:
                self._current_row_is_header = True
            attrs_dict = dict(attrs)
            try:
                self._cell_rowspan = max(1, int(attrs_dict.get("rowspan", "1") or "1"))
            except ValueError:
                self._cell_rowspan = 1
            try:
                self._cell_colspan = max(1, int(attrs_dict.get("colspan", "1") or "1"))
            except ValueError:
                self._cell_colspan = 1
            self._cell_buffer = []
        elif t == "br" and self._in_cell:
            self._cell_buffer.append(" ")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._in_cell:
            text = "".join(self._cell_buffer).strip()
            # Collapse internal whitespace
            text = re.sub(r"\s+", " ", text)
            # Defer col assignment — done when row ends (depends on pending rowspans)
            self._current_row_cells.append(
                (len(self._current_row_cells), self._cell_rowspan, self._cell_colspan,
                 text, self._cell_is_header)
            )
            self._in_cell = False
            self._cell_rowspan = 1
            self._cell_colspan = 1
            self._cell_buffer = []
        elif t == "tr" and self._in_row:
            self._finalize_row()
            self._in_row = False
        elif t == "thead":
            self._in_thead = False
        elif t == "tbody":
            self._in_tbody = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buffer.append(data)

    def _finalize_row(self) -> None:
        """Assign column indices to cells in current row, honoring pending rowspans."""
        # Build the expanded row: walk column by column, consuming cells
        expanded: list[str] = []
        cell_iter = iter(self._current_row_cells)
        col = 0
        next_cell = next(cell_iter, None)

        while True:
            # First, fill columns occupied by pending rowspans from previous rows
            if col in self._pending_rowspans:
                remaining, text = self._pending_rowspans[col]
                expanded.append(text)
                # Decrement; remove if exhausted after this row
                if remaining - 1 <= 0:
                    del self._pending_rowspans[col]
                else:
                    self._pending_rowspans[col] = (remaining - 1, text)
                col += 1
                continue

            # Else, place the next cell from this row
            if next_cell is None:
                break

            _orig_idx, rs, cs, text, _is_header = next_cell
            # Place `text` in cs consecutive cols
            for offset in range(cs):
                expanded.append(text)
                target_col = col + offset
                # Register rowspan in pending for future rows
                if rs > 1:
                    self._pending_rowspans[target_col] = (rs - 1, text)
            col += cs
            next_cell = next(cell_iter, None)

        # Even if no cells in row, pending rowspans at the end should be filled
        # (tail of row). Check for pending cols beyond what we've filled:
        while col in self._pending_rowspans:
            remaining, text = self._pending_rowspans[col]
            expanded.append(text)
            if remaining - 1 <= 0:
                del self._pending_rowspans[col]
            else:
                self._pending_rowspans[col] = (remaining - 1, text)
            col += 1

        # Track max cols
        if len(expanded) > self.n_cols:
            self.n_cols = len(expanded)

        # Emit
        if self._current_row_is_header or self._in_thead:
            self.header_rows.append(expanded)
        else:
            # Body row — emit cells
            for c_idx, text in enumerate(expanded):
                self.body_cells.append(Cell(
                    row=self._body_row_idx,
                    col=c_idx,
                    text=text,
                    is_header=False,
                ))
            self._body_row_idx += 1

        self._current_row_cells = []
        self._current_row_is_header = False


def parse_single_table(html: str, page: int) -> Table:
    """Parse one <table>...</table> HTML block into a Table object."""
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        # Malformed HTML — return empty table with raw_html preserved
        return Table(page=page, n_rows=0, n_cols=0, raw_html=html)

    # Normalize header row widths to n_cols
    n_cols = parser.n_cols
    headers = [
        row + [""] * (n_cols - len(row)) if len(row) < n_cols else row
        for row in parser.header_rows
    ]

    n_rows = max((c.row for c in parser.body_cells), default=-1) + 1

    return Table(
        page=page,
        n_rows=n_rows,
        n_cols=n_cols,
        headers=headers,
        cells=parser.body_cells,
        raw_html=html,
    )


def parse_tables_from_markdown(markdown: str, page: int) -> list[Table]:
    """Extract every <table>...</table> block from markdown, parse each."""
    tables: list[Table] = []
    for _start, _end, html in find_table_blocks(markdown):
        tables.append(parse_single_table(html, page))
    return tables
