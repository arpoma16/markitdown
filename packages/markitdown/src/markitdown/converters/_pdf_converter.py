import sys
import io
import os
import re
from typing import BinaryIO, Any

from .._base_converter import DocumentConverter, DocumentConverterResult
from .._stream_info import StreamInfo
from .._exceptions import MissingDependencyException, MISSING_DEPENDENCY_MESSAGE

# Pattern for MasterFormat-style partial numbering (e.g., ".1", ".2", ".10")
PARTIAL_NUMBERING_PATTERN = re.compile(r"^\.\d+$")


def _merge_partial_numbering_lines(text: str) -> str:
    """
    Post-process extracted text to merge MasterFormat-style partial numbering
    with the following text line.

    MasterFormat documents use partial numbering like:
        .1  The intent of this Request for Proposal...
        .2  Available information relative to...

    Some PDF extractors split these into separate lines:
        .1
        The intent of this Request for Proposal...

    This function merges them back together.
    """
    lines = text.split("\n")
    result_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this line is ONLY a partial numbering
        if PARTIAL_NUMBERING_PATTERN.match(stripped):
            # Look for the next non-empty line to merge with
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            if j < len(lines):
                # Merge the partial numbering with the next line
                next_line = lines[j].strip()
                result_lines.append(f"{stripped} {next_line}")
                i = j + 1  # Skip past the merged line
            else:
                # No next line to merge with, keep as is
                result_lines.append(line)
                i += 1
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


# Load dependencies
_dependency_exc_info = None
try:
    import pdfminer
    import pdfminer.high_level
    import pdfplumber
except ImportError:
    _dependency_exc_info = sys.exc_info()


ACCEPTED_MIME_TYPE_PREFIXES = [
    "application/pdf",
    "application/x-pdf",
]

ACCEPTED_FILE_EXTENSIONS = [".pdf"]


def _find_column_split(page: Any) -> float | None:
    """
    Detect pages laid out as two text columns, e.g. scientific papers, and
    return the x position of the gutter (None if the page is not multi-column).

    A page is considered multi-column when a vertical gutter near the middle of
    the page is crossed by almost no words, and both sides of the gutter hold a
    substantial amount of prose-like text (many reasonably long lines). Such
    pages must be read column by column; row-based extraction interleaves them.
    """
    try:
        words = page.extract_words()
    except Exception:
        return None
    if len(words) < 150:
        return None

    page_width = float(getattr(page, "width", 612))
    lo, hi = page_width * 0.35, page_width * 0.65
    total = len(words)

    # Sweep candidate gutter positions across the central band and keep the
    # ones crossed by (almost) no words. Full-width titles/captions are few.
    max_crossing = max(3, int(total * 0.03))
    step = 2.0
    gutter: list[float] = []
    x = lo
    while x <= hi:
        crossing = sum(1 for w in words if w["x0"] < x < w["x1"])
        if crossing <= max_crossing:
            gutter.append(x)
        x += step

    # Require a contiguous empty channel at least ~6pt wide
    if not gutter or (gutter[-1] - gutter[0]) < 6:
        return None
    split_x = (gutter[0] + gutter[-1]) / 2

    def prose_lines(side: list[dict]) -> int:
        lines: dict[int, list[dict]] = {}
        for w in side:
            lines.setdefault(round(w["top"] / 3), []).append(w)
        return sum(
            1
            for ws in lines.values()
            if sum(len(w["text"]) for w in ws) >= 25 and len(ws) >= 4
        )

    left = [w for w in words if w["x1"] <= split_x]
    right = [w for w in words if w["x0"] >= split_x]
    if prose_lines(left) >= 10 and prose_lines(right) >= 10:
        return split_x
    return None


def _looks_glued(text: str) -> bool:
    """True when extracted text has (almost) no spaces between words."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 200:
        return False
    return text.count(" ") / len(chars) < 0.08


def _extract_columns_text(
    page: Any, split_x: float, images: list[dict] | None = None
) -> str:
    """
    Read a two-column page column by column from pdfplumber's words.

    Full-width rows (titles, spanning captions) are emitted in place; between
    them the left column is read completely before the right one. Images
    (with their "link") are placed at their position in that same order. Used
    when pdfminer cannot see the gaps between words (PDFs without space
    glyphs).
    """
    words = page.extract_words(x_tolerance=1.0)
    words.sort(key=lambda w: (w["top"], w["x0"]))

    # Cluster words into visual rows to spot full-width rows
    rows: list[list[dict]] = []
    for w in words:
        if rows and abs(w["top"] - rows[-1][0]["top"]) <= 3:
            rows[-1].append(w)
        else:
            rows.append([w])

    def is_full_width(row: list[dict]) -> bool:
        if any(w["x0"] < split_x < w["x1"] for w in row):
            return True
        left = [w["x1"] for w in row if w["x1"] <= split_x]
        right = [w["x0"] for w in row if w["x0"] >= split_x]
        # Centered titles break between words close to the gutter
        return bool(left and right and min(right) - max(left) < 10)

    def lines_of(ws: list[dict]) -> list[str]:
        ws = sorted(ws, key=lambda w: (w["top"], w["x0"]))
        out: list[list[dict]] = []
        for w in ws:
            # An image is always a line of its own
            if (
                out
                and not w.get("img")
                and not out[-1][0].get("img")
                and abs(w["top"] - out[-1][0]["top"]) <= 3
            ):
                out[-1].append(w)
            else:
                out.append([w])
        return [
            " ".join(x["text"] for x in sorted(line, key=lambda w: w["x0"]))
            for line in out
        ]

    # Rows and images in top-to-bottom order
    events: list[tuple[float, str, Any]] = [(r[0]["top"], "row", r) for r in rows]
    for im in images or []:
        events.append((im["top"], "img", im))
    events.sort(key=lambda e: e[0])

    result: list[str] = []
    left_buf: list[dict] = []
    right_buf: list[dict] = []

    def flush() -> None:
        result.extend(lines_of(left_buf))
        result.extend(lines_of(right_buf))
        left_buf.clear()
        right_buf.clear()

    for _, kind, obj in events:
        if kind == "row":
            if is_full_width(obj):
                flush()
                result.extend(lines_of(obj))
            else:
                for w in obj:
                    (left_buf if w["x1"] <= split_x else right_buf).append(w)
        elif obj["x0"] < split_x < obj["x1"]:
            flush()
            result.append(obj["link"])
        else:
            pseudo = {
                "text": obj["link"],
                "x0": obj["x0"],
                "x1": obj["x1"],
                "top": obj["top"],
                "img": True,
            }
            center = (obj["x0"] + obj["x1"]) / 2
            (left_buf if center < split_x else right_buf).append(pseudo)
    flush()
    return "\n".join(result)


def _to_markdown_table(table: list[list[str]], include_separator: bool = True) -> str:
    """Convert a 2D list (rows/columns) into a nicely aligned Markdown table.

    Args:
        table: 2D list of cell values
        include_separator: If True, include header separator row (standard markdown).
                          If False, output simple pipe-separated rows.
    """
    if not table:
        return ""

    # Normalize None → ""
    table = [[cell if cell is not None else "" for cell in row] for row in table]

    # Filter out empty rows
    table = [row for row in table if any(cell.strip() for cell in row)]

    if not table:
        return ""

    # Column widths
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]

    def fmt_row(row: list[str]) -> str:
        return (
            "|"
            + "|".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
            + "|"
        )

    if include_separator:
        header, *rows = table
        md = [fmt_row(header)]
        md.append("|" + "|".join("-" * w for w in col_widths) + "|")
        for row in rows:
            md.append(fmt_row(row))
    else:
        md = [fmt_row(row) for row in table]

    return "\n".join(md)


def _extract_form_content_from_words(page: Any) -> str | None:
    """
    Extract form-style content from a PDF page by analyzing word positions.
    This handles borderless forms/tables where words are aligned in columns.

    Returns markdown with proper table formatting:
    - Tables have pipe-separated columns with header separator rows
    - Non-table content is rendered as plain text

    Returns None if the page doesn't appear to be a form-style document,
    indicating that pdfminer should be used instead for better text spacing.
    """
    words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
    if not words:
        return None

    # Group words by their Y position (rows)
    y_tolerance = 5
    rows_by_y: dict[float, list[dict]] = {}
    for word in words:
        y_key = round(word["top"] / y_tolerance) * y_tolerance
        if y_key not in rows_by_y:
            rows_by_y[y_key] = []
        rows_by_y[y_key].append(word)

    # Sort rows by Y position
    sorted_y_keys = sorted(rows_by_y.keys())
    page_width = page.width if hasattr(page, "width") else 612

    # First pass: analyze each row
    row_info: list[dict] = []
    for y_key in sorted_y_keys:
        row_words = sorted(rows_by_y[y_key], key=lambda w: w["x0"])
        if not row_words:
            continue

        first_x0 = row_words[0]["x0"]
        last_x1 = row_words[-1]["x1"]
        line_width = last_x1 - first_x0
        combined_text = " ".join(w["text"] for w in row_words)

        # Count distinct x-position groups (columns)
        x_positions = [w["x0"] for w in row_words]
        x_groups: list[float] = []
        for x in sorted(x_positions):
            if not x_groups or x - x_groups[-1] > 50:
                x_groups.append(x)

        # Determine row type
        is_paragraph = line_width > page_width * 0.55 and len(combined_text) > 60

        # Check for MasterFormat-style partial numbering (e.g., ".1", ".2")
        # These should be treated as list items, not table rows
        has_partial_numbering = False
        if row_words:
            first_word = row_words[0]["text"].strip()
            if PARTIAL_NUMBERING_PATTERN.match(first_word):
                has_partial_numbering = True

        row_info.append(
            {
                "y_key": y_key,
                "words": row_words,
                "text": combined_text,
                "x_groups": x_groups,
                "is_paragraph": is_paragraph,
                "num_columns": len(x_groups),
                "has_partial_numbering": has_partial_numbering,
            }
        )

    # Collect ALL x-positions from rows with 3+ columns (table-like rows)
    # This gives us the global column structure
    all_table_x_positions: list[float] = []
    for info in row_info:
        if info["num_columns"] >= 3 and not info["is_paragraph"]:
            all_table_x_positions.extend(info["x_groups"])

    if not all_table_x_positions:
        return None

    # Compute adaptive column clustering tolerance based on gap analysis
    all_table_x_positions.sort()

    # Calculate gaps between consecutive x-positions
    gaps = []
    for i in range(len(all_table_x_positions) - 1):
        gap = all_table_x_positions[i + 1] - all_table_x_positions[i]
        if gap > 5:  # Only significant gaps
            gaps.append(gap)

    # Determine optimal tolerance using statistical analysis
    if gaps and len(gaps) >= 3:
        # Use 70th percentile of gaps as threshold (balances precision/recall)
        sorted_gaps = sorted(gaps)
        percentile_70_idx = int(len(sorted_gaps) * 0.70)
        adaptive_tolerance = sorted_gaps[percentile_70_idx]

        # Clamp tolerance to reasonable range [25, 50]
        adaptive_tolerance = max(25, min(50, adaptive_tolerance))
    else:
        # Fallback to conservative value
        adaptive_tolerance = 35

    # Compute global column boundaries using adaptive tolerance
    global_columns: list[float] = []
    for x in all_table_x_positions:
        if not global_columns or x - global_columns[-1] > adaptive_tolerance:
            global_columns.append(x)

    # Adaptive max column check based on page characteristics
    # Calculate average column width
    if len(global_columns) > 1:
        content_width = global_columns[-1] - global_columns[0]
        avg_col_width = content_width / len(global_columns)

        # Forms with very narrow columns (< 30px) are likely dense text
        if avg_col_width < 30:
            return None

        # Compute adaptive max based on columns per inch
        # Typical forms have 3-8 columns per inch
        columns_per_inch = len(global_columns) / (content_width / 72)

        # If density is too high (> 10 cols/inch), likely not a form
        if columns_per_inch > 10:
            return None

        # Adaptive max: allow more columns for wider pages
        # Standard letter is 612pt wide, so scale accordingly
        adaptive_max_columns = int(20 * (page_width / 612))
        adaptive_max_columns = max(15, adaptive_max_columns)  # At least 15

        if len(global_columns) > adaptive_max_columns:
            return None
    else:
        # Single column, not a form
        return None

    # Now classify each row as table row or not
    # A row is a table row if it has words that align with 2+ of the global columns
    for info in row_info:
        if info["is_paragraph"]:
            info["is_table_row"] = False
            continue

        # Rows with partial numbering (e.g., ".1", ".2") are list items, not table rows
        if info["has_partial_numbering"]:
            info["is_table_row"] = False
            continue

        # Count how many global columns this row's words align with
        aligned_columns: set[int] = set()
        for word in info["words"]:
            word_x = word["x0"]
            for col_idx, col_x in enumerate(global_columns):
                if abs(word_x - col_x) < 40:
                    aligned_columns.add(col_idx)
                    break

        # If row uses 2+ of the established columns, it's a table row
        info["is_table_row"] = len(aligned_columns) >= 2

    # Find table regions (consecutive table rows)
    table_regions: list[tuple[int, int]] = []  # (start_idx, end_idx)
    i = 0
    while i < len(row_info):
        if row_info[i]["is_table_row"]:
            start_idx = i
            while i < len(row_info) and row_info[i]["is_table_row"]:
                i += 1
            end_idx = i
            table_regions.append((start_idx, end_idx))
        else:
            i += 1

    # Check if enough rows are table rows (at least 20%)
    total_table_rows = sum(end - start for start, end in table_regions)
    if len(row_info) > 0 and total_table_rows / len(row_info) < 0.2:
        return None

    # Build output - collect table data first, then format with proper column widths
    result_lines: list[str] = []
    num_cols = len(global_columns)

    # Helper function to extract cells from a row
    def extract_cells(info: dict) -> list[str]:
        cells: list[str] = ["" for _ in range(num_cols)]
        for word in info["words"]:
            word_x = word["x0"]
            # Find the correct column using boundary ranges
            assigned_col = num_cols - 1  # Default to last column
            for col_idx in range(num_cols - 1):
                col_end = global_columns[col_idx + 1]
                if word_x < col_end - 20:
                    assigned_col = col_idx
                    break
            if cells[assigned_col]:
                cells[assigned_col] += " " + word["text"]
            else:
                cells[assigned_col] = word["text"]
        return cells

    # Process rows, collecting table data for proper formatting
    idx = 0
    while idx < len(row_info):
        info = row_info[idx]

        # Check if this row starts a table region
        table_region = None
        for start, end in table_regions:
            if idx == start:
                table_region = (start, end)
                break

        if table_region:
            start, end = table_region
            # Collect all rows in this table
            table_data: list[list[str]] = []
            for table_idx in range(start, end):
                cells = extract_cells(row_info[table_idx])
                table_data.append(cells)

            # Calculate column widths for this table
            if table_data:
                col_widths = [
                    max(len(row[col]) for row in table_data) for col in range(num_cols)
                ]
                # Ensure minimum width of 3 for separator dashes
                col_widths = [max(w, 3) for w in col_widths]

                # Format header row
                header = table_data[0]
                header_str = (
                    "| "
                    + " | ".join(
                        cell.ljust(col_widths[i]) for i, cell in enumerate(header)
                    )
                    + " |"
                )
                result_lines.append(header_str)

                # Format separator row
                separator = (
                    "| "
                    + " | ".join("-" * col_widths[i] for i in range(num_cols))
                    + " |"
                )
                result_lines.append(separator)

                # Format data rows
                for row in table_data[1:]:
                    row_str = (
                        "| "
                        + " | ".join(
                            cell.ljust(col_widths[i]) for i, cell in enumerate(row)
                        )
                        + " |"
                    )
                    result_lines.append(row_str)

            idx = end  # Skip to end of table region
        else:
            # Check if we're inside a table region (not at start)
            in_table = False
            for start, end in table_regions:
                if start < idx < end:
                    in_table = True
                    break

            if not in_table:
                # Non-table content
                result_lines.append(info["text"])
            idx += 1

    return "\n".join(result_lines)


def _extract_tables_from_words(page: Any) -> list[list[list[str]]]:
    """
    Extract tables from a PDF page by analyzing word positions.
    This handles borderless tables where words are aligned in columns.

    This function is designed for structured tabular data (like invoices),
    not for multi-column text layouts in scientific documents.
    """
    words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
    if not words:
        return []

    # Group words by their Y position (rows)
    y_tolerance = 5
    rows_by_y: dict[float, list[dict]] = {}
    for word in words:
        y_key = round(word["top"] / y_tolerance) * y_tolerance
        if y_key not in rows_by_y:
            rows_by_y[y_key] = []
        rows_by_y[y_key].append(word)

    # Sort rows by Y position
    sorted_y_keys = sorted(rows_by_y.keys())

    # Find potential column boundaries by analyzing x positions across all rows
    all_x_positions = []
    for words_in_row in rows_by_y.values():
        for word in words_in_row:
            all_x_positions.append(word["x0"])

    if not all_x_positions:
        return []

    # Cluster x positions to find column starts
    all_x_positions.sort()
    x_tolerance_col = 20
    column_starts: list[float] = []
    for x in all_x_positions:
        if not column_starts or x - column_starts[-1] > x_tolerance_col:
            column_starts.append(x)

    # Need at least 3 columns but not too many (likely text layout, not table)
    if len(column_starts) < 3 or len(column_starts) > 10:
        return []

    # Find rows that span multiple columns (potential table rows)
    table_rows = []
    for y_key in sorted_y_keys:
        words_in_row = sorted(rows_by_y[y_key], key=lambda w: w["x0"])

        # Assign words to columns
        row_data = [""] * len(column_starts)
        for word in words_in_row:
            # Find the closest column
            best_col = 0
            min_dist = float("inf")
            for i, col_x in enumerate(column_starts):
                dist = abs(word["x0"] - col_x)
                if dist < min_dist:
                    min_dist = dist
                    best_col = i

            if row_data[best_col]:
                row_data[best_col] += " " + word["text"]
            else:
                row_data[best_col] = word["text"]

        # Only include rows that have content in multiple columns
        non_empty = sum(1 for cell in row_data if cell.strip())
        if non_empty >= 2:
            table_rows.append(row_data)

    # Validate table quality - tables should have:
    # 1. Enough rows (at least 3 including header)
    # 2. Short cell content (tables have concise data, not paragraphs)
    # 3. Consistent structure across rows
    if len(table_rows) < 3:
        return []

    # Check if cells contain short, structured data (not long text)
    long_cell_count = 0
    total_cell_count = 0
    for row in table_rows:
        for cell in row:
            if cell.strip():
                total_cell_count += 1
                # If cell has more than 30 chars, it's likely prose text
                if len(cell.strip()) > 30:
                    long_cell_count += 1

    # If more than 30% of cells are long, this is probably not a table
    if total_cell_count > 0 and long_cell_count / total_cell_count > 0.3:
        return []

    return [table_rows]


def _save_page_images(
    page: Any, page_idx: int, images_dir: str, stem: str, min_size: float = 50.0
) -> list[dict]:
    """
    Save the embedded raster images of a page into `images_dir`. Returns, for
    each image (top-to-bottom, left-to-right), its page position
    (x0, x1, top, bottom) and its Markdown link under "link".

    Images are rendered from their bounding box on the page (so any filter or
    colorspace is handled), at their native resolution. Tiny images (icons,
    rules) are skipped.
    """
    found: list[dict] = []
    try:
        images = sorted(page.images, key=lambda im: (im["top"], im["x0"]))
    except Exception:
        return found

    x0p, y0p, x1p, y1p = page.bbox
    n = 0
    for im in images:
        width, height = im["x1"] - im["x0"], im["bottom"] - im["top"]
        if width < min_size or height < min_size:
            continue
        bbox = (
            max(im["x0"], x0p),
            max(im["top"], y0p),
            min(im["x1"], x1p),
            min(im["bottom"], y1p),
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        resolution = 150.0
        srcsize = im.get("srcsize")
        if srcsize and srcsize[0]:
            # Match the image's native pixel density, within sane bounds
            resolution = min(300.0, max(72.0, srcsize[0] / (width / 72.0)))

        n += 1
        name = f"{stem}_p{page_idx + 1}_img{n}.png"
        try:
            os.makedirs(images_dir, exist_ok=True)
            page.crop(bbox).to_image(resolution=resolution).save(
                os.path.join(images_dir, name)
            )
        except Exception:
            n -= 1
            continue
        found.append(
            {
                "x0": bbox[0],
                "x1": bbox[2],
                "top": bbox[1],
                "bottom": bbox[3],
                "link": f"![{stem} page {page_idx + 1} image {n}]"
                f"({os.path.join(images_dir, name)})",
            }
        )
    return found


def _reading_key(
    x0: float, x1: float, cy: float, split_x: float, separators: list[float]
) -> tuple[int, int, float]:
    """
    Sort key (top-down y: cy grows downwards) for a two-column page.

    Full-width items (crossing the gutter) split the page into bands; inside a
    band the left column is read before the right one.
    """
    above = sum(1 for sep in separators if sep < cy)
    if x0 < split_x < x1:
        return (above, 2, cy)
    return (above, 0 if (x0 + x1) / 2 < split_x else 1, cy)


def _pdfminer_page_with_images(
    pdf_bytes: Any, page_idx: int, page_height: float, split_x: float, images: list[dict]
) -> str:
    """pdfminer's text boxes for one two-column page, with images in place."""
    from pdfminer.layout import LTTextContainer

    items: list[tuple[float, float, float, str]] = []  # x0, x1, cy(top-down), text
    pdf_bytes.seek(0)
    for layout in pdfminer.high_level.extract_pages(pdf_bytes, page_numbers=[page_idx]):
        for obj in layout:
            if isinstance(obj, LTTextContainer):
                text = obj.get_text().strip()
                if text:
                    cy = page_height - (obj.y0 + obj.y1) / 2
                    items.append((obj.x0, obj.x1, cy, text))
    for im in images:
        items.append((im["x0"], im["x1"], (im["top"] + im["bottom"]) / 2, im["link"]))

    separators = [cy for x0, x1, cy, _ in items if x0 < split_x < x1]
    items.sort(key=lambda it: _reading_key(it[0], it[1], it[2], split_x, separators))
    return "\n\n".join(text for *_, text in items)


def _plain_page_with_images(page: Any, images: list[dict]) -> str:
    """Single-column page text with images placed at their vertical position."""
    items: list[tuple[float, str]] = []
    try:
        for line in page.extract_text_lines():
            items.append((line["top"], line["text"]))
    except Exception:
        items = [(0.0, page.extract_text() or "")]
    for im in images:
        items.append(((im["top"] + im["bottom"]) / 2, "\n" + im["link"] + "\n"))
    items.sort(key=lambda it: it[0])
    return "\n".join(text for _, text in items)


class PdfConverter(DocumentConverter):
    """
    Converts PDFs to Markdown.
    Supports extracting tables into aligned Markdown format (via pdfplumber).
    Falls back to pdfminer if pdfplumber is missing or fails.
    """

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in ACCEPTED_FILE_EXTENSIONS:
            return True

        for prefix in ACCEPTED_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True

        return False

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        if _dependency_exc_info is not None:
            raise MissingDependencyException(
                MISSING_DEPENDENCY_MESSAGE.format(
                    converter=type(self).__name__,
                    extension=".pdf",
                    feature="pdf",
                )
            ) from _dependency_exc_info[1].with_traceback(
                _dependency_exc_info[2]
            )  # type: ignore[union-attr]

        assert isinstance(file_stream, io.IOBase)

        # Read file stream into BytesIO for compatibility with pdfplumber
        pdf_bytes = io.BytesIO(file_stream.read())

        try:
            # Single pass: check every page for form-style content.
            # Pages with tables/forms get rich extraction; plain-text
            # pages are collected separately. page.close() is called
            # after each page to free pdfplumber's cached objects and
            # keep memory usage constant regardless of page count.
            images_dir = kwargs.get("images_dir")
            stem = "document"
            if stream_info.filename:
                stem = os.path.splitext(os.path.basename(stream_info.filename))[0]
            positional_pages: dict[int, str] = {}
            markdown_chunks: list[str] = []
            form_page_count = 0
            multi_column_page_count = 0
            pdfminer_pages: list[str] | None = None

            with pdfplumber.open(pdf_bytes) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    images = (
                        _save_page_images(page, page_idx, images_dir, stem)
                        if images_dir
                        else []
                    )

                    # Multi-column prose (e.g. papers) is read correctly by
                    # pdfminer's layout analysis; row-based extraction
                    # interleaves the columns.
                    split_x = _find_column_split(page)
                    if split_x is not None:
                        if pdfminer_pages is None:
                            pdf_bytes.seek(0)
                            pdfminer_pages = pdfminer.high_level.extract_text(
                                pdf_bytes
                            ).split("\f")
                        text = (
                            pdfminer_pages[page_idx]
                            if page_idx < len(pdfminer_pages)
                            else ""
                        )
                        # Some PDFs have no space glyphs and pdfminer glues the
                        # words together; pdfplumber's word boxes still work.
                        if _looks_glued(text):
                            text = _extract_columns_text(page, split_x, images)
                        elif images:
                            text = _pdfminer_page_with_images(
                                pdf_bytes, page_idx, float(page.height), split_x, images
                            )
                        multi_column_page_count += 1
                        if text.strip():
                            markdown_chunks.append(text.strip())
                        page.close()
                        continue

                    page_content = _extract_form_content_from_words(page)

                    if page_content is not None:
                        form_page_count += 1
                        if page_content.strip():
                            markdown_chunks.append(page_content)
                        # Table layout is rebuilt from words, so images of a
                        # form page go after its content.
                        if images:
                            markdown_chunks.append(
                                "\n\n".join(im["link"] for im in images)
                            )
                    else:
                        if images:
                            text = _plain_page_with_images(page, images)
                            positional_pages[page_idx] = text
                        else:
                            text = page.extract_text()
                        if text and text.strip():
                            markdown_chunks.append(text.strip())

                    page.close()  # Free cached page data immediately

            # If no pages had form-style or multi-column content, use pdfminer
            # for the whole document (better text spacing for prose).
            if form_page_count == 0 and multi_column_page_count == 0:
                pdf_bytes.seek(0)
                markdown = pdfminer.high_level.extract_text(pdf_bytes)
                if positional_pages:
                    # Pages with images keep their images in place
                    pages = markdown.split("\f")
                    for idx, text in positional_pages.items():
                        if idx < len(pages):
                            pages[idx] = text
                    markdown = "\n\n".join(p.strip() for p in pages)
            else:
                markdown = "\n\n".join(markdown_chunks).strip()

        except Exception:
            # Fallback if pdfplumber fails
            pdf_bytes.seek(0)
            markdown = pdfminer.high_level.extract_text(pdf_bytes)

        # Fallback if still empty
        if not markdown:
            pdf_bytes.seek(0)
            markdown = pdfminer.high_level.extract_text(pdf_bytes)

        # Post-process to merge MasterFormat-style partial numbering with following text
        markdown = _merge_partial_numbering_lines(markdown)

        return DocumentConverterResult(markdown=markdown)
