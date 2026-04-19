"""Typed dataclasses for parsed document structure."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Cell:
    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


@dataclass
class Table:
    """Parsed table with resolved span structure."""
    page: int
    n_rows: int
    n_cols: int
    headers: list[list[str]] = field(default_factory=list)
    cells: list[Cell] = field(default_factory=list)
    title: str = ""
    caption: str = ""
    raw_html: str = ""

    def flat_headers(self) -> list[str]:
        """Flatten multi-row headers into compound strings ('Parent > Child')."""
        if not self.headers:
            return [""] * self.n_cols
        flat: list[str] = []
        for col in range(self.n_cols):
            parts: list[str] = []
            for row in self.headers:
                if col < len(row):
                    val = row[col].strip()
                    if val and (not parts or parts[-1] != val):
                        parts.append(val)
            flat.append(" > ".join(parts))
        return flat

    def body_rows(self) -> list[list[str]]:
        """Return body cells as grid (list of rows, each a list of cell texts)."""
        if not self.cells:
            return []
        max_row = max(c.row for c in self.cells) if self.cells else 0
        rows: list[list[str]] = [[""] * self.n_cols for _ in range(max_row + 1)]
        for c in self.cells:
            if 0 <= c.row <= max_row and 0 <= c.col < self.n_cols:
                rows[c.row][c.col] = c.text
        return rows


@dataclass
class Figure:
    page: int
    bbox: tuple[float, float, float, float] | None = None
    caption: str = ""
    alt_text: str = ""


@dataclass
class Section:
    id: str
    title: str
    level: int
    page_start: int
    page_end: int
    summary: str = ""


@dataclass
class Document:
    """Top-level structured document."""
    name: str
    total_pages: int
    sections: list[Section] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    page_markdowns: dict[int, str] = field(default_factory=dict)
    ellipsis_pages: dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_pages": self.total_pages,
            "sections": [s.__dict__ for s in self.sections],
            "tables": [
                {
                    "page": t.page,
                    "n_rows": t.n_rows,
                    "n_cols": t.n_cols,
                    "headers": t.headers,
                    "flat_headers": t.flat_headers(),
                    "body": t.body_rows(),
                    "title": t.title,
                    "caption": t.caption,
                }
                for t in self.tables
            ],
            "figures": [f.__dict__ for f in self.figures],
            "ellipsis_pages": self.ellipsis_pages,
        }
