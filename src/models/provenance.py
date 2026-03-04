"""
ProvenanceChain — the audit trail linking every extracted fact to its
exact location in the source document.

Week 3's spatial provenance mirrors Week 1's content_hash concept:
a spatially-addressed record that remains valid even when document
content is reformatted or re-ordered.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ProvenanceRecord(BaseModel):
    """
    A single source citation — one LDU's origin in the physical document.
    """
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int = Field(ge=1)
    bounding_box: Optional[dict] = Field(
        default=None,
        description="Serialised BoundingBox {x0,y0,x1,y1,page}",
    )
    content_hash: str = Field(
        description="SHA-256 of the chunk content — used to verify the citation has not drifted"
    )
    excerpt: str = Field(
        description="Short verbatim excerpt (≤200 chars) from the source chunk"
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Parent section heading for human-readable navigation",
    )


class ProvenanceChain(BaseModel):
    """
    The complete provenance attached to one answer / extracted fact.

    Every answer produced by the Query Interface Agent must carry a
    ProvenanceChain.  An answer without provenance is treated as
    unverifiable and flagged accordingly.
    """
    query: str = Field(description="The original user question or extraction request")
    answer: str = Field(description="The answer / extracted fact")
    sources: list[ProvenanceRecord] = Field(
        description="Ordered list of source citations (most relevant first)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Aggregate confidence of the answer given the source evidence",
    )
    is_verified: bool = Field(
        default=False,
        description="True if the answer has been verified against source citations in Audit Mode",
    )
    unverifiable_flag: bool = Field(
        default=False,
        description="True if no supporting source was found — claim cannot be verified",
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
