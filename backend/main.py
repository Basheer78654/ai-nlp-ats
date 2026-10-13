"""FastAPI application for resume screening and candidate ranking."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from spacy.tokens import Span

from extractor import (
    estimate_experience_years,
    extract_contact_info,
    extract_skills,
    load_nlp,
)
from matcher import ATSMatcher
from parser import extract_text_from_docx, extract_text_from_pdf, segment_resume_sections
from vector_store import CandidateVectorStore

_EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
_PHONE_PATTERN: re.Pattern[str] = re.compile(
    r"(?<!\w)(?:\+|00)?\d(?:[ \t().-]*\d){6,14}(?!\w)"
)
_DATE_RANGE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present|current|now|today)\b",
    re.IGNORECASE,
)

app: FastAPI = FastAPI(title="AI-Powered ATS Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher: ATSMatcher = ATSMatcher()
candidate_store: CandidateVectorStore = CandidateVectorStore()


class RankRequest(BaseModel):
    """Inputs for ranking candidates already indexed in the vector store."""

    job_description: str
    required_skills: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1)


class AnonymizeRequest(BaseModel):
    """Text to anonymize."""

    text: str


def _parse_resume(filename: str, file_bytes: bytes) -> str:
    extension: str = Path(filename).suffix.casefold()
    if extension == ".pdf":
        text: str = extract_text_from_pdf(file_bytes)
    elif extension == ".docx":
        text = extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported resume file type. Upload a PDF or DOCX file.",
        )
    if not text.strip():
        raise HTTPException(status_code=422, detail="The uploaded resume contains no text.")
    return text


def _extract_candidate_data(text: str) -> tuple[dict[str, str | None], list[str], float]:
    sections: dict[str, str] = segment_resume_sections(text)
    experience_text: str = sections["experience"] or text
    return (
        extract_contact_info(text),
        extract_skills(text),
        estimate_experience_years(experience_text),
    )


def _skills_from_metadata(metadata: dict[str, Any]) -> list[str]:
    stored_skills: Any = metadata.get("skills", [])
    if isinstance(stored_skills, str):
        try:
            stored_skills = json.loads(stored_skills)
        except json.JSONDecodeError:
            stored_skills = [skill.strip() for skill in stored_skills.split(",")]
    if not isinstance(stored_skills, list):
        return []
    return [skill for skill in stored_skills if isinstance(skill, str)]


def _anonymize_text(text: str) -> str:
    spans: list[tuple[int, int, str]] = [
        (match.start(), match.end(), "[REDACTED_EMAIL]")
        for match in _EMAIL_PATTERN.finditer(text)
    ]
    date_ranges: list[tuple[int, int]] = [
        match.span() for match in _DATE_RANGE_PATTERN.finditer(text)
    ]
    for match in _PHONE_PATTERN.finditer(text):
        digit_count: int = sum(character.isdigit() for character in match.group(0))
        if not 7 <= digit_count <= 15:
            continue
        if any(
            match.start() < range_end and match.end() > range_start
            for range_start, range_end in date_ranges
        ):
            continue
        spans.append((match.start(), match.end(), "[REDACTED_PHONE]"))

    document = load_nlp()(text)
    person_spans: list[Span] = [
        entity for entity in document.ents if entity.label_ == "PERSON"
    ]
    for entity in person_spans:
        spans.append((entity.start_char, entity.end_char, "[REDACTED_NAME]"))

    selected_spans: list[tuple[int, int, str]] = []
    for start, end, replacement in sorted(spans, key=lambda span: (span[0], -(span[1] - span[0]))):
        if selected_spans and start < selected_spans[-1][1]:
            continue
        selected_spans.append((start, end, replacement))

    anonymized_text: str = text
    for start, end, replacement in reversed(selected_spans):
        anonymized_text = f"{anonymized_text[:start]}{replacement}{anonymized_text[end:]}"
    return anonymized_text


@app.get("/health")
async def health() -> dict[str, str | int]:
    indexed_resumes: int = await asyncio.to_thread(candidate_store.count)
    return {"status": "ok", "indexed_resumes": indexed_resumes}


@app.post("/api/screen-single")
async def screen_single(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    required_skills: str = Form(""),
) -> dict[str, Any]:
    filename: str = resume.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="The uploaded resume must have a filename.")
    file_bytes: bytes = await resume.read()
    resume_text: str = await asyncio.to_thread(_parse_resume, filename, file_bytes)
    contact_details, candidate_skills, experience_years = await asyncio.to_thread(
        _extract_candidate_data,
        resume_text,
    )
    skills: list[str] = [
        skill.strip() for skill in required_skills.split(",") if skill.strip()
    ]
    metrics: dict[str, Any] = await asyncio.to_thread(
        matcher.match_candidate,
        candidate_skills,
        resume_text,
        job_description,
        skills,
        experience_years,
    )
    return {
        "contact_details": contact_details,
        "extracted_skills": candidate_skills,
        **metrics,
    }


@app.post("/api/upload-bulk")
async def upload_bulk(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    indexed_candidates: list[dict[str, str]] = []
    for uploaded_file in files:
        filename: str = uploaded_file.filename or ""
        if not filename:
            raise HTTPException(status_code=400, detail="Every uploaded resume must have a filename.")
        file_bytes: bytes = await uploaded_file.read()
        resume_text: str = await asyncio.to_thread(_parse_resume, filename, file_bytes)
        contact_details, candidate_skills, experience_years = await asyncio.to_thread(
            _extract_candidate_data,
            resume_text,
        )
        candidate_id: str = str(uuid.uuid4())
        await asyncio.to_thread(
            candidate_store.add_resume,
            candidate_id,
            resume_text,
            {
                "candidate_name": contact_details["name"] or "",
                "skills": candidate_skills,
                "experience_years": experience_years,
                "original_filename": filename,
            },
        )
        indexed_candidates.append(
            {
                "candidate_id": candidate_id,
                "filename": filename,
                "candidate_name": contact_details["name"] or "",
            }
        )
    return {"indexed_count": len(indexed_candidates), "candidates": indexed_candidates}


@app.post("/api/rank-pool")
async def rank_pool(request: RankRequest) -> dict[str, list[dict[str, Any]]]:
    matches: list[dict[str, Any]] = await asyncio.to_thread(
        candidate_store.search_candidates,
        request.job_description,
        request.top_k,
    )
    leaderboard: list[dict[str, Any]] = []
    for candidate in matches:
        metadata: dict[str, Any] = candidate["metadata"]
        candidate_skills: list[str] = _skills_from_metadata(metadata)
        experience_value: Any = metadata.get("experience_years", 0.0)
        try:
            experience_years: float = float(experience_value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid stored experience value for candidate {candidate['candidate_id']!r}."
            ) from error

        metrics: dict[str, Any] = await asyncio.to_thread(
            matcher.match_candidate,
            candidate_skills,
            candidate["document"],
            request.job_description,
            request.required_skills,
            experience_years,
        )
        leaderboard.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_name": metadata.get("candidate_name", ""),
                "skills": candidate_skills,
                "distance": candidate["distance"],
                **metrics,
            }
        )
    leaderboard.sort(key=lambda candidate: candidate["overall_score"], reverse=True)
    return {"leaderboard": leaderboard}


@app.post("/api/anonymize-text")
async def anonymize_text(request: AnonymizeRequest) -> dict[str, str]:
    anonymized_text: str = await asyncio.to_thread(_anonymize_text, request.text)
    return {"anonymized_text": anonymized_text}
