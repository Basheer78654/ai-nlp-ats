"""Persistent ChromaDB storage for candidate resumes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import ChromaError


def _metadata_scalar(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value
    return None


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}

    def add_value(key: str, value: Any) -> None:
        if not key:
            return
        if value is None:
            return
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                if not str(nested_key):
                    continue
                add_value(f"{key}.{nested_key}", nested_value)
            return

        scalar: str | int | float | bool | None = _metadata_scalar(value)
        if scalar is not None:
            sanitized[key] = scalar
        elif isinstance(value, (list, tuple, set, frozenset)):
            list_value: list[Any] = list(value)
            sanitized[key] = json.dumps(list_value, default=str, separators=(",", ":"))
        else:
            sanitized[key] = str(value)

    for metadata_key, metadata_value in metadata.items():
        add_value(str(metadata_key), metadata_value)
    return sanitized


class CandidateVectorStore:
    """Store and search resume documents in a persistent ChromaDB collection."""

    def __init__(self) -> None:
        database_path: Path = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
        try:
            self.client: chromadb.ClientAPI = chromadb.PersistentClient(path=str(database_path))
            self.collection: chromadb.Collection = self.client.get_or_create_collection(
                name="resumes"
            )
        except ChromaError as error:
            raise RuntimeError("Failed to initialize the candidate vector store.") from error

    def add_resume(self, candidate_id: str, text: str, metadata: dict[str, Any]) -> None:
        """Add or replace a candidate resume and its sanitized metadata."""
        if not candidate_id.strip():
            raise ValueError("candidate_id must not be empty.")
        if not text.strip():
            raise ValueError("Resume text must not be empty.")

        sanitized_metadata: dict[str, str | int | float | bool] = _sanitize_metadata(
            metadata
        )
        sanitized_metadata["candidate_id"] = candidate_id
        try:
            self.collection.upsert(
                ids=[candidate_id],
                documents=[text],
                metadatas=[sanitized_metadata],
            )
        except ChromaError as error:
            raise RuntimeError(f"Failed to store resume for candidate {candidate_id!r}.") from error

    def search_candidates(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the nearest resume documents and their metadata."""
        if not query_text.strip():
            raise ValueError("query_text must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        try:
            results: dict[str, Any] = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except ChromaError as error:
            raise RuntimeError("Failed to search candidate resumes.") from error

        ids: list[list[str]] = results["ids"]
        documents: list[list[str | None]] | None = results["documents"]
        metadatas: list[list[dict[str, Any] | None]] | None = results["metadatas"]
        distances: list[list[float]] | None = results["distances"]
        if not ids or documents is None or metadatas is None or distances is None:
            raise RuntimeError("ChromaDB returned an incomplete search result.")
        if not documents or not metadatas or not distances:
            raise RuntimeError("ChromaDB returned an incomplete search result.")
        if not (
            len(ids[0])
            == len(documents[0])
            == len(metadatas[0])
            == len(distances[0])
        ):
            raise RuntimeError("ChromaDB returned inconsistent search result lengths.")

        matches: list[dict[str, Any]] = []
        for index, candidate_id in enumerate(ids[0]):
            document: str | None = documents[0][index]
            metadata: dict[str, Any] | None = metadatas[0][index]
            distance: float = distances[0][index]
            if document is None or metadata is None:
                raise RuntimeError("ChromaDB returned a candidate without its document or metadata.")
            matches.append(
                {
                    "candidate_id": candidate_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return matches

    def delete_candidate(self, candidate_id: str) -> None:
        """Delete a candidate resume by its ID."""
        if not candidate_id.strip():
            raise ValueError("candidate_id must not be empty.")
        try:
            self.collection.delete(ids=[candidate_id])
        except ChromaError as error:
            raise RuntimeError(f"Failed to delete candidate {candidate_id!r}.") from error

    def count(self) -> int:
        """Return the number of resumes in the collection."""
        try:
            return self.collection.count()
        except ChromaError as error:
            raise RuntimeError("Failed to count candidate resumes.") from error
