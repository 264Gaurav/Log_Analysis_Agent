"""Retrieval diagnostics shared by the hybrid retriever and analysis nodes."""

from dataclasses import dataclass, field
import json
import os
from typing import Any, Iterable, Sequence

from logger import fingerprint


@dataclass
class FusedDocument:
    """A document and its component retrieval contributions."""

    document: Any
    score: float = 0.0
    contributions: dict[str, float] = field(default_factory=dict)


def document_id(document: Any) -> str:
    """Build a stable, content-safe identifier for a retrieved chunk."""
    metadata = getattr(document, "metadata", {}) or {}
    content = getattr(document, "page_content", "")
    return fingerprint(json.dumps({"metadata": metadata, "content": content}, sort_keys=True, default=str))


def document_details(document: Any, rank: int | None = None, score: float | None = None) -> dict[str, Any]:
    """Return useful chunk metadata without logging content by default."""
    content = getattr(document, "page_content", "") or ""
    metadata = getattr(document, "metadata", {}) or {}
    details: dict[str, Any] = {
        "chunk_id": document_id(document),
        "chunk_chars": len(content),
        "chunk_sha256": fingerprint(content),
        "source": metadata.get("source"),
        "start_index": metadata.get("start_index"),
    }
    if rank is not None:
        details["rank"] = rank
    if score is not None:
        details["raw_score"] = round(float(score), 6)
    preview_length = int(os.getenv("LOG_CONTENT_PREVIEW_CHARS", "0"))
    if preview_length > 0:
        details["chunk_preview"] = content[:preview_length].replace("\n", "\\n")
    return details


def rrf_fuse(
    ranked_results: Sequence[tuple[str, Sequence[Any], Sequence[float | None]]],
    weights: Sequence[float],
    rank_constant: int = 60,
) -> list[FusedDocument]:
    """Fuse ranked component results and retain per-retriever RRF contributions."""
    fused: dict[str, FusedDocument] = {}
    for (retriever_name, documents, scores), weight in zip(ranked_results, weights):
        for rank, document in enumerate(documents, start=1):
            chunk_id = document_id(document)
            item = fused.setdefault(chunk_id, FusedDocument(document=document))
            contribution = weight / (rank_constant + rank)
            item.score += contribution
            item.contributions[retriever_name] = contribution
    return sorted(fused.values(), key=lambda item: item.score, reverse=True)


def log_component_results(logger: Any, retriever_name: str, documents: Iterable[Any], scores: Sequence[float | None]) -> None:
    """Log the component ranking matrix one candidate at a time."""
    for rank, (document, score) in enumerate(zip(documents, scores), start=1):
        details = document_details(document, rank=rank, score=score)
        logger.info(
            "stage=retrieve event=component_candidate retriever=%s details=%r",
            retriever_name,
            details,
        )


def log_fused_results(logger: Any, fused_documents: Sequence[FusedDocument]) -> None:
    """Log final RRF scores and each returned chunk."""
    for rank, item in enumerate(fused_documents, start=1):
        details = document_details(item.document, rank=rank, score=item.score)
        logger.info(
            "stage=retrieve event=rrf_candidate rank=%d rrf_score=%.8f contributions=%r details=%r",
            rank,
            item.score,
            item.contributions,
            details,
        )
