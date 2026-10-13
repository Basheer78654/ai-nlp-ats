"""Score resume fit against a job description."""

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class ATSMatcher:
    """Combine semantic similarity, required skills, and keyword relevance."""

    def __init__(self) -> None:
        self.model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")

    def compute_dense_similarity(self, resume_text: str, jd_text: str) -> float:
        """Return cosine similarity scaled to the range [0.0, 1.0]."""
        embeddings: np.ndarray = self.model.encode(
            [resume_text, jd_text],
            normalize_embeddings=True,
        )
        cosine_similarity: float = float(np.dot(embeddings[0], embeddings[1]))
        return float(np.clip(cosine_similarity, 0.0, 1.0))

    def compute_sparse_score(
        self,
        resume_tokens: list[str],
        jd_tokens: list[str],
    ) -> float:
        """Return BM25 keyword relevance normalized to the range [0.0, 1.0]."""
        normalized_resume_tokens: list[str] = [
            token.casefold() for token in resume_tokens if token.strip()
        ]
        normalized_jd_tokens: list[str] = [
            token.casefold() for token in jd_tokens if token.strip()
        ]
        if not normalized_resume_tokens or not normalized_jd_tokens:
            return 0.0

        # Extra empty documents keep one-resume BM25 IDF values non-negative.
        bm25: BM25Okapi = BM25Okapi(
            [normalized_resume_tokens, [], [], []]
        )
        raw_score: float = float(bm25.get_scores(normalized_jd_tokens)[0])
        if raw_score <= 0.0:
            return 0.0
        return raw_score / (1.0 + raw_score)

    def match_candidate(
        self,
        candidate_skills: list[str],
        resume_text: str,
        jd_text: str,
        required_skills: list[str],
        detected_experience: float = 0.0,
    ) -> dict[str, Any]:
        """Compute weighted candidate fit and report matched/missing skills."""
        candidate_skill_keys: set[str] = {
            skill.strip().casefold() for skill in candidate_skills if skill.strip()
        }
        unique_required_skills: dict[str, str] = {}
        for skill in required_skills:
            skill_key: str = skill.strip().casefold()
            if skill_key:
                unique_required_skills.setdefault(skill_key, skill)

        matched_skills: list[str] = [
            skill
            for key, skill in unique_required_skills.items()
            if key in candidate_skill_keys
        ]
        missing_skills: list[str] = [
            skill
            for key, skill in unique_required_skills.items()
            if key not in candidate_skill_keys
        ]
        skill_match_percentage: float = (
            len(matched_skills) / len(unique_required_skills) * 100.0
            if unique_required_skills
            else 0.0
        )

        semantic_score: float = self.compute_dense_similarity(resume_text, jd_text)
        resume_tokens: list[str] = resume_text.split()
        jd_tokens: list[str] = jd_text.split()
        bm25_score: float = self.compute_sparse_score(resume_tokens, jd_tokens)
        overall_score: float = (
            0.5 * semantic_score
            + 0.3 * (skill_match_percentage / 100.0)
            + 0.2 * bm25_score
        ) * 100.0

        return {
            "overall_score": overall_score,
            "semantic_score": semantic_score,
            "skill_match_percentage": skill_match_percentage,
            "bm25_score": bm25_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "detected_experience_years": detected_experience,
        }
