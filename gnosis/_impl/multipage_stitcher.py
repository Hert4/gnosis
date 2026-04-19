"""Multi-page table stitching — heuristic Azure-style.

Detects when a table continues on the next page and merges body rows.
Rules (conservative — favor false negative over false positive):

  (1) Pages are strictly consecutive (N, N+1).
  (2) Page N has at least one parsed Table; page N+1 has at least one.
  (3) Last table of N and first table of N+1 have identical n_cols.
  (4) Headers match: either both flat_headers lists equal, OR both are
      effectively empty (all cells empty strings).

When a merge happens, the second table's header row is dropped (it's a
repeated header) and its body rows are appended to the first table's body.
Spans of 3+ pages are handled by iteratively applying the 2-page rule.

Does NOT call any LLM. Pure Python, deterministic.
"""

from __future__ import annotations

from dataclasses import replace

from gnosis._impl.native_schema import Cell, Table


def _headers_match(a: list[str], b: list[str]) -> bool:
    """True if two flat header lists are effectively equal."""
    if len(a) != len(b):
        return False
    a_stripped = [h.strip() for h in a]
    b_stripped = [h.strip() for h in b]
    if a_stripped == b_stripped:
        return True
    # Both effectively empty (all cells blank) → still considered a match
    if all(not h for h in a_stripped) and all(not h for h in b_stripped):
        return True
    return False


def can_merge(prev: Table, curr: Table) -> bool:
    """Whether two tables on consecutive pages should be stitched."""
    if prev.n_cols == 0 or curr.n_cols == 0:
        return False
    if prev.n_cols != curr.n_cols:
        return False
    return _headers_match(prev.flat_headers(), curr.flat_headers())


def merge_two(prev: Table, curr: Table) -> Table:
    """Concatenate body rows of curr into prev (drop curr's header).

    Returned Table has:
      - page set to prev.page (continuation anchored at start page)
      - headers from prev (curr's header is dropped, assumed duplicate)
      - cells = prev.cells + curr.cells with row indices offset
      - n_rows = prev.n_rows + curr.n_rows
    """
    row_offset = prev.n_rows
    shifted_cells: list[Cell] = [
        replace(c, row=c.row + row_offset) for c in curr.cells
    ]
    merged_cells = list(prev.cells) + shifted_cells

    return Table(
        page=prev.page,
        n_rows=prev.n_rows + curr.n_rows,
        n_cols=prev.n_cols,
        headers=prev.headers,
        cells=merged_cells,
        title=prev.title or curr.title,
        caption=prev.caption or curr.caption,
        raw_html=prev.raw_html,  # preserve first page HTML reference
    )


def detect_spans(
    page_tables: dict[int, list[Table]],
) -> list[list[int]]:
    """Find runs of consecutive pages whose boundary tables can merge.

    Returns list of page-number groups, each with 2+ pages. E.g.
    [[603, 604, 605], [611, 612, 613, 614]].
    """
    pages = sorted(page_tables.keys())
    spans: list[list[int]] = []
    current: list[int] = []

    for i, pg in enumerate(pages):
        if not current:
            current = [pg]
            continue

        prev_pg = current[-1]
        if pg != prev_pg + 1:
            # Non-consecutive — close current span
            if len(current) >= 2:
                spans.append(current)
            current = [pg]
            continue

        # Check boundary tables
        prev_tables = page_tables.get(prev_pg, [])
        curr_tables = page_tables.get(pg, [])
        if not prev_tables or not curr_tables:
            if len(current) >= 2:
                spans.append(current)
            current = [pg]
            continue

        if can_merge(prev_tables[-1], curr_tables[0]):
            current.append(pg)
        else:
            if len(current) >= 2:
                spans.append(current)
            current = [pg]

    if len(current) >= 2:
        spans.append(current)
    return spans


def stitch_span(
    page_tables: dict[int, list[Table]],
    span: list[int],
) -> Table:
    """Merge the boundary tables across a span into one Table."""
    assert len(span) >= 2, "span must cover at least 2 pages"

    first_pg = span[0]
    merged = page_tables[first_pg][-1]  # start from last table of first page
    for pg in span[1:]:
        nxt = page_tables[pg][0]  # first table of next page
        merged = merge_two(merged, nxt)
    return merged


def stitch_document(
    page_tables: dict[int, list[Table]],
) -> tuple[dict[int, list[Table]], list[dict]]:
    """Apply stitching across the whole document.

    Returns:
        (new_page_tables, merge_report)

    new_page_tables: for each span, the merged table REPLACES the last
        table of the first span page. Tables on subsequent span pages
        that participated in the merge are removed (their boundary table
        is absorbed into the previous merge).
    merge_report: list of {spans_pages, n_cols, total_rows_after,
        rows_from_each_page}
    """
    spans = detect_spans(page_tables)
    if not spans:
        return dict(page_tables), []

    new_tables: dict[int, list[Table]] = {k: list(v) for k, v in page_tables.items()}
    report: list[dict] = []

    for span in spans:
        rows_per_page = [page_tables[pg][0 if pg != span[0] else -1].n_rows for pg in span]
        merged = stitch_span(page_tables, span)

        # Write merged back: replace last table of first page
        first_pg = span[0]
        first_page_tables = new_tables[first_pg]
        if first_page_tables:
            first_page_tables[-1] = merged
        else:
            new_tables[first_pg] = [merged]

        # Remove absorbed boundary tables from subsequent pages
        for pg in span[1:]:
            if new_tables.get(pg):
                # drop the first table (which was just absorbed)
                new_tables[pg] = new_tables[pg][1:]

        report.append({
            "spans_pages": span,
            "n_cols": merged.n_cols,
            "total_rows_after": merged.n_rows,
            "rows_from_each_page": rows_per_page,
            "headers_empty": all(not h.strip() for h in merged.flat_headers()),
        })

    return new_tables, report
