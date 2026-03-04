"""
Logical Document Unit (LDU) — the atomic chunk emitted by the ChunkingEngine.

An LDU is the unit of retrieval: it is self-contained, semantically
coherent, and carries enough metadata to reconstruct its provenance
precisely (page, bounding-box, section ancestry, content hash).
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ChunkType(str, Enum):
    """The semantic type of a chunk, used to apply type-specific retrieval rules."""
    paragraph = "paragraph"
    heading = "heading"
    table = "table"
    table_row = "table_row"
    figure = "figure"
    list_block = "list_block"
    caption = "caption"
    footnote = "footnote"
    header_footer = "header_footer"
    equation = "equation"
    other = "other"


class ChunkRelationship(BaseModel):
    """A typed relationship to another LDU in the same document."""
    target_chunk_id: str
    relation_type: str = Field(
        description="parent | child | cross_ref | next | prev | figure_of | table_of"
    )


class LDU(BaseModel):
    """
    Logical Document Unit — the RAG-ready atomic chunk.

    Chunking rules enforced at creation (see ChunkingEngine):
    1. A table cell is never split from its header row  → chunk_type=table carries all rows
    2. Figure caption is stored as metadata of its parent figure chunk
    3. A numbered list is kept as a single LDU unless it exceeds max_tokens
    4. Section headers are stored as parent_section on all child chunks
    5. Cross-references ("see Table 3") are resolved into relationships
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    chunk_id: str = Field(description="Globally unique ID: {doc_id}_{sequence:06d}")
    doc_id: str
    sequence: int = Field(ge=0, description="Zero-based position in the document's chunk list")

    # ── Content ───────────────────────────────────────────────────────────────
    content: str = Field(description="Rendered text content of this chunk")
    chunk_type: ChunkType
    token_count: int = Field(ge=0)

    # ── Spatial provenance ────────────────────────────────────────────────────
    page_refs: list[int] = Field(
        description="1-indexed page numbers this chunk spans (may be >1 for multi-page tables)"
    )
    bounding_box: Optional[dict] = Field(
        default=None,
        description="Serialised BoundingBox dict {x0,y0,x1,y1,page} for the primary page",
    )

    # ── Structural context ────────────────────────────────────────────────────
    parent_section: Optional[str] = Field(
        default=None,
        description="Title of the enclosing section heading (rule 4)",
    )
    section_depth: int = Field(default=0, ge=0, description="Nesting depth of parent_section")
    relationships: list[ChunkRelationship] = Field(default_factory=list)

    # ── Integrity ─────────────────────────────────────────────────────────────
    content_hash: str = Field(
        description="SHA-256 of the UTF-8 encoded content — enables provenance verification"
    )

    @model_validator(mode="before")
    @classmethod
    def compute_content_hash(cls, data: dict) -> dict:
        if "content_hash" not in data or not data.get("content_hash"):
            content = data.get("content", "")
            data["content_hash"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return data

    class Config:
        use_enum_values = True
