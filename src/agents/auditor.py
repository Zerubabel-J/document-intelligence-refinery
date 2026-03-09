"""
Provenance & Audit System — Audit Mode.

Given a claim (e.g. "The report states revenue was 4.2B in Q3"), the
system either:
  - VERIFIES with a source citation (ProvenanceChain), or
  - FLAGS as "not found / unverifiable".

Verification process:
  1. Search for supporting evidence in the vector store + fact table
  2. Compare claim against source content using text similarity
  3. Build a ProvenanceChain with page references and bounding boxes
  4. Set is_verified=True if evidence confidence > threshold
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Optional

from src.models.ldu import LDU
from src.models.provenance import ProvenanceChain, ProvenanceRecord
from src.storage.fact_table import FactTableDB
from src.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

_VERIFICATION_THRESHOLD = 0.65


class AuditEngine:
    """
    Claim verification engine with full provenance support.

    Usage::

        engine = AuditEngine(vector_store=vs, fact_db=db)
        chain = engine.verify_claim(
            claim="Revenue was 4.2 billion birr in FY 2023",
            doc_id="abc123",
        )
        print(chain.is_verified)   # True / False
        print(chain.sources)       # ProvenanceRecord list
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        fact_db: Optional[FactTableDB] = None,
        verification_threshold: float = _VERIFICATION_THRESHOLD,
    ) -> None:
        self._vs = vector_store
        self._db = fact_db
        self._threshold = verification_threshold

    def verify_claim(
        self,
        claim: str,
        doc_id: Optional[str] = None,
    ) -> ProvenanceChain:
        """
        Verify a claim against stored evidence.

        Returns a ProvenanceChain with is_verified or unverifiable_flag set.
        """
        sources: list[ProvenanceRecord] = []
        confidence_scores: list[float] = []

        # ── Source 1: Vector store semantic search ────────────────────────────
        if self._vs:
            vs_results = self._vs.search(claim, top_k=5, doc_id=doc_id)
            for hit in vs_results:
                sim = 1.0 - hit.get("distance", 1.0)  # cosine distance → similarity
                if sim > 0.3:  # minimum relevance threshold
                    metadata = hit.get("metadata", {})
                    page_refs_str = metadata.get("page_refs", "1")
                    page = int(page_refs_str.split(",")[0]) if page_refs_str else 1
                    content = hit.get("content", "")

                    sources.append(ProvenanceRecord(
                        chunk_id=hit.get("chunk_id", ""),
                        doc_id=metadata.get("doc_id", doc_id or ""),
                        doc_name=metadata.get("doc_name", ""),
                        page_number=page,
                        content_hash=metadata.get("content_hash", hashlib.sha256(content.encode()).hexdigest()),
                        excerpt=content[:200],
                        section_title=metadata.get("parent_section"),
                    ))
                    confidence_scores.append(sim)

        # ── Source 2: Fact table structured search ────────────────────────────
        if self._db:
            # Extract keywords from claim for fact search
            keywords = self._extract_keywords(claim)
            for kw in keywords:
                facts = self._db.search_facts(kw, doc_id=doc_id)
                for fact in facts[:3]:
                    fact_text = f"{fact['key']}: {fact['value']}"
                    sim = self._text_similarity(claim, fact_text)
                    if sim > 0.2:
                        sources.append(ProvenanceRecord(
                            chunk_id=fact.get("chunk_id", ""),
                            doc_id=fact.get("doc_id", doc_id or ""),
                            doc_name=fact.get("doc_name", ""),
                            page_number=fact.get("page_number", 1) or 1,
                            content_hash=hashlib.sha256(fact_text.encode()).hexdigest(),
                            excerpt=fact_text[:200],
                            section_title=fact.get("section"),
                        ))
                        confidence_scores.append(sim)

        # ── Compute aggregate confidence ──────────────────────────────────────
        if confidence_scores:
            aggregate_conf = max(confidence_scores)
        else:
            aggregate_conf = 0.0

        is_verified = aggregate_conf >= self._threshold
        unverifiable = len(sources) == 0

        # Deduplicate sources by chunk_id
        seen: set[str] = set()
        unique_sources: list[ProvenanceRecord] = []
        for src in sources:
            key = f"{src.chunk_id}_{src.page_number}"
            if key not in seen:
                seen.add(key)
                unique_sources.append(src)

        return ProvenanceChain(
            query=claim,
            answer=self._format_answer(claim, unique_sources, is_verified),
            sources=unique_sources[:5],  # cap at 5 citations
            confidence=round(aggregate_conf, 4),
            is_verified=is_verified,
            unverifiable_flag=unverifiable,
        )

    def build_provenance_for_answer(
        self,
        query: str,
        answer: str,
        supporting_chunks: list[dict],
    ) -> ProvenanceChain:
        """
        Build a ProvenanceChain for an answer produced by the Query Agent.

        supporting_chunks: list of dicts with keys: chunk_id, content, metadata.
        """
        sources: list[ProvenanceRecord] = []
        for chunk in supporting_chunks:
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")
            page_refs_str = metadata.get("page_refs", "1")
            page = int(page_refs_str.split(",")[0]) if page_refs_str else 1

            sources.append(ProvenanceRecord(
                chunk_id=chunk.get("chunk_id", ""),
                doc_id=metadata.get("doc_id", ""),
                doc_name=metadata.get("doc_name", ""),
                page_number=page,
                content_hash=metadata.get("content_hash", hashlib.sha256(content.encode()).hexdigest()),
                excerpt=content[:200],
                section_title=metadata.get("parent_section"),
            ))

        return ProvenanceChain(
            query=query,
            answer=answer,
            sources=sources,
            confidence=0.85 if sources else 0.0,
            is_verified=bool(sources),
            unverifiable_flag=not bool(sources),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract important keywords from a claim for fact table search."""
        # Remove common words
        stopwords = {
            "the", "a", "an", "is", "was", "were", "are", "be", "been",
            "in", "of", "to", "for", "on", "at", "by", "with", "from",
            "that", "this", "it", "its", "and", "or", "but", "not",
            "report", "states", "says", "shows", "indicates",
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]
        # Also extract numbers
        numbers = re.findall(r'[\d,]+(?:\.\d+)?', text)
        return (keywords + numbers)[:5]

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """Simple word-overlap Jaccard similarity."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _format_answer(claim: str, sources: list[ProvenanceRecord], verified: bool) -> str:
        if not sources:
            return f"UNVERIFIABLE: No supporting evidence found for claim: '{claim}'"
        status = "VERIFIED" if verified else "PARTIALLY SUPPORTED"
        source_refs = ", ".join(
            f"p.{s.page_number} ({s.section_title or 'unknown section'})"
            for s in sources[:3]
        )
        return f"[{status}] {claim} — Sources: {source_refs}"
