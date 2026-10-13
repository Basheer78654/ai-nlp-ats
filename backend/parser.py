"""Parse unstructured resume PDFs and DOCX files into text and sections."""

from __future__ import annotations

import io
import re
from typing import Final

import pymupdf as fitz
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

RESUME_SECTIONS: Final[tuple[str, ...]] = (
    "summary",
    "skills",
    "experience",
    "education",
    "certifications",
)

_SPAN_GAP_PX: Final[float] = 2.0

_SECTION_HEADER_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "summary": re.compile(
        r"^(?:professional\s+|career\s+|executive\s+)?"
        r"(?:summary|profile|objective|overview|about(?:\s+me)?)"
        r"(?:\s+of\s+qualifications)?\s*:?\s*$",
        re.IGNORECASE,
    ),
    "skills": re.compile(
        r"^(?:core\s+|key\s+|technical\s+|professional\s+)?"
        r"(?:skills|skill\s+set|competencies|technologies|tech\s+stack|"
        r"tools(?:\s+and\s+technologies)?)\s*:?\s*$",
        re.IGNORECASE,
    ),
    "experience": re.compile(
        r"^(?:professional\s+|relevant\s+|work\s+)?"
        r"(?:experience|employment(?:\s+history)?|work\s+history|"
        r"career\s+history)\s*:?\s*$",
        re.IGNORECASE,
    ),
    "education": re.compile(
        r"^(?:academic\s+)?"
        r"(?:education|academics|educational\s+background|"
        r"academic\s+background)\s*:?\s*$",
        re.IGNORECASE,
    ),
    "certifications": re.compile(
        r"^(?:licenses?\s*(?:and|&)\s*)?"
        r"(?:certifications?|certificates?|licenses?|licences?)"
        r"(?:\s*(?:and|&)\s*certifications?)?\s*:?\s*$",
        re.IGNORECASE,
    ),
}

_LEADING_JUNK: Final[re.Pattern[str]] = re.compile(
    r"^[\s•●○■▪►\-–—*·]+"
)
_TRAILING_COLON: Final[re.Pattern[str]] = re.compile(r"\s*:+\s*$")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract PDF text with block and line layout preserved."""
    if not file_bytes:
        return ""

    pages: list[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        for page in document:
            page_text: str = _extract_pdf_page(page)
            if page_text:
                pages.append(page_text)

    return "\n\n".join(pages).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract DOCX text from paragraphs and tables in document order."""
    if not file_bytes:
        return ""

    document: Document = Document(io.BytesIO(file_bytes))
    chunks: list[str] = []

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            paragraph_text: str = Paragraph(child, document).text.strip()
            if paragraph_text:
                chunks.append(paragraph_text)
        elif isinstance(child, CT_Tbl):
            table_text: str = _table_to_text(Table(child, document))
            if table_text:
                chunks.append(table_text)

    return "\n".join(chunks).strip()


def segment_resume_sections(text: str) -> dict[str, str]:
    """Split resume text into canonical sections using header heuristics."""
    sections: dict[str, str] = {key: "" for key in RESUME_SECTIONS}
    if not text or not text.strip():
        return sections

    lines: list[str] = text.splitlines()
    current_key: str | None = None
    buckets: dict[str, list[str]] = {key: [] for key in RESUME_SECTIONS}
    preamble: list[str] = []

    for raw_line in lines:
        header_key: str | None = _match_section_header(raw_line)
        if header_key is not None:
            current_key = header_key
            continue
        if current_key is None:
            preamble.append(raw_line)
            continue
        buckets[current_key].append(raw_line)

    for key in RESUME_SECTIONS:
        sections[key] = _join_section_lines(buckets[key])

    if not sections["summary"]:
        sections["summary"] = _join_section_lines(preamble)

    return sections


def _extract_pdf_page(page: fitz.Page) -> str:
    payload: dict[str, object] = page.get_text("dict")
    raw_blocks: object = payload.get("blocks", [])
    if not isinstance(raw_blocks, list):
        return ""

    text_blocks: list[tuple[tuple[float, float, float, float], str]] = []
    for block in raw_blocks:
        if not isinstance(block, dict) or block.get("type", 0) != 0:
            continue
        bbox_obj: object = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        bbox: tuple[float, float, float, float] = _as_bbox(bbox_obj)
        block_text: str = _extract_pdf_block_lines(block)
        if block_text:
            text_blocks.append((bbox, block_text))

    text_blocks.sort(key=lambda item: (round(item[0][1], 1), round(item[0][0], 1)))
    return "\n\n".join(block_text for _, block_text in text_blocks)


def _extract_pdf_block_lines(block: dict[str, object]) -> str:
    raw_lines: object = block.get("lines", [])
    if not isinstance(raw_lines, list):
        return ""

    lines: list[tuple[tuple[float, float, float, float], str]] = []
    for line in raw_lines:
        if not isinstance(line, dict):
            continue
        line_text: str = _join_pdf_spans(line.get("spans", []))
        if not line_text:
            continue
        bbox: tuple[float, float, float, float] = _as_bbox(line.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        lines.append((bbox, line_text))

    lines.sort(key=lambda item: (round(item[0][1], 1), round(item[0][0], 1)))
    return "\n".join(text for _, text in lines)


def _join_pdf_spans(raw_spans: object) -> str:
    if not isinstance(raw_spans, list):
        return ""

    parts: list[str] = []
    previous_x1: float | None = None
    for span in raw_spans:
        if not isinstance(span, dict):
            continue
        text: str = str(span.get("text", ""))
        if not text:
            continue
        bbox: tuple[float, float, float, float] = _as_bbox(span.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if previous_x1 is not None and bbox[0] - previous_x1 > _SPAN_GAP_PX:
            if not parts[-1].endswith(" ") and not text.startswith(" "):
                parts.append(" ")
        parts.append(text)
        previous_x1 = bbox[2]

    return "".join(parts).strip()


def _as_bbox(value: object) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return (0.0, 0.0, 0.0, 0.0)


def _table_to_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        seen_tc_ids: set[int] = set()
        for cell in row.cells:
            tc_id: int = id(cell._tc)
            if tc_id in seen_tc_ids:
                continue
            seen_tc_ids.add(tc_id)
            cell_text: str = " ".join(paragraph.text.strip() for paragraph in cell.paragraphs if paragraph.text.strip())
            if cell_text:
                cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _match_section_header(line: str) -> str | None:
    candidate: str = _normalize_header_line(line)
    if not candidate or not _looks_like_header(candidate):
        return None
    for key, pattern in _SECTION_HEADER_PATTERNS.items():
        if pattern.match(candidate):
            return key
    return None


def _normalize_header_line(line: str) -> str:
    stripped: str = line.strip()
    stripped = _LEADING_JUNK.sub("", stripped)
    stripped = _TRAILING_COLON.sub("", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def _looks_like_header(candidate: str) -> bool:
    if len(candidate) > 60 or len(candidate.split()) > 6:
        return False
    letters: str = re.sub(r"[^A-Za-z]", "", candidate)
    if len(letters) < 4:
        return False
    is_short: bool = len(candidate) <= 40
    is_all_caps: bool = candidate.upper() == candidate and any(ch.isalpha() for ch in candidate)
    is_title_case: bool = candidate == candidate.title()
    return is_short or is_all_caps or is_title_case


def _join_section_lines(lines: list[str]) -> str:
    joined: str = "\n".join(line.rstrip() for line in lines).strip()
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined
