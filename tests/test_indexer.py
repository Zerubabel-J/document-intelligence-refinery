"""
Unit tests for the PageIndex Builder.

Tests cover:
1. Tree construction from heading LDUs
2. Section assignment of content chunks
3. Navigation / search over the tree
4. Entity extraction
"""

from __future__ import annotations

import pytest

from src.agents.indexer import PageIndexBuilder
from src.models.ldu import LDU, ChunkType


def _ldu(seq: int, content: str, chunk_type: ChunkType, page: int = 1,
         parent_section: str | None = None, depth: int = 0) -> LDU:
    import hashlib
    return LDU(
        chunk_id=f"doc_{seq:06d}",
        doc_id="doc",
        sequence=seq,
        content=content,
        chunk_type=chunk_type,
        token_count=len(content.split()),
        page_refs=[page],
        parent_section=parent_section,
        section_depth=depth,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


@pytest.fixture()
def builder():
    return PageIndexBuilder(enable_llm_summaries=False)


class TestPageIndexConstruction:

    def test_builds_tree_from_headings(self, builder):
        ldus = [
            _ldu(0, "Introduction", ChunkType.heading, page=1, depth=1),
            _ldu(1, "This is the intro text.", ChunkType.paragraph, page=1, parent_section="Introduction"),
            _ldu(2, "Methodology", ChunkType.heading, page=3, depth=1),
            _ldu(3, "We used pdfplumber.", ChunkType.paragraph, page=3, parent_section="Methodology"),
            _ldu(4, "Results", ChunkType.heading, page=5, depth=1),
            _ldu(5, "Revenue grew.", ChunkType.paragraph, page=5, parent_section="Results"),
        ]
        index = builder.build(ldus, doc_id="doc", doc_name="test.pdf")
        assert index.total_sections == 3
        assert len(index.root_nodes) == 3
        assert index.root_nodes[0].title == "Introduction"
        assert index.root_nodes[1].title == "Methodology"

    def test_nested_sections(self, builder):
        ldus = [
            _ldu(0, "Chapter 1", ChunkType.heading, page=1, depth=1),
            _ldu(1, "Section 1.1", ChunkType.heading, page=2, depth=2),
            _ldu(2, "Content here.", ChunkType.paragraph, page=2, parent_section="Section 1.1"),
            _ldu(3, "Section 1.2", ChunkType.heading, page=4, depth=2),
            _ldu(4, "More content.", ChunkType.paragraph, page=4, parent_section="Section 1.2"),
        ]
        index = builder.build(ldus, doc_id="doc", doc_name="test.pdf")
        assert len(index.root_nodes) == 1
        assert index.root_nodes[0].title == "Chapter 1"
        assert len(index.root_nodes[0].child_nodes) == 2

    def test_chunks_assigned_to_sections(self, builder):
        ldus = [
            _ldu(0, "Revenue Analysis", ChunkType.heading, page=1, depth=1),
            _ldu(1, "Total revenue was 4.2B.", ChunkType.paragraph, page=1, parent_section="Revenue Analysis"),
        ]
        index = builder.build(ldus, doc_id="doc", doc_name="test.pdf")
        node = index.root_nodes[0]
        assert "doc_000001" in node.chunk_ids

    def test_no_headings_creates_single_root(self, builder):
        ldus = [
            _ldu(0, "Just a paragraph.", ChunkType.paragraph, page=1),
            _ldu(1, "Another one.", ChunkType.paragraph, page=2),
        ]
        index = builder.build(ldus, doc_id="doc", doc_name="test.pdf")
        assert index.total_sections == 1
        assert index.root_nodes[0].title == "Document"


class TestPageIndexNavigation:

    def test_navigate_finds_relevant_section(self, builder):
        ldus = [
            _ldu(0, "Financial Overview", ChunkType.heading, page=1, depth=1),
            _ldu(1, "Revenue was 4.2B.", ChunkType.paragraph, page=1, parent_section="Financial Overview"),
            _ldu(2, "Appendix", ChunkType.heading, page=10, depth=1),
            _ldu(3, "Extra details.", ChunkType.paragraph, page=10, parent_section="Appendix"),
        ]
        index = builder.build(ldus, doc_id="doc", doc_name="test.pdf")
        results = index.navigate("Financial", top_k=1)
        assert len(results) == 1
        assert results[0].title == "Financial Overview"

    def test_navigate_returns_empty_for_no_match(self, builder):
        ldus = [
            _ldu(0, "Chapter 1", ChunkType.heading, page=1, depth=1),
        ]
        index = builder.build(ldus, doc_id="doc", doc_name="test.pdf")
        results = index.navigate("nonexistent topic xyz", top_k=3)
        assert len(results) == 0


class TestFallbackSummary:

    def test_fallback_summary_from_paragraphs(self, builder):
        ldus = [
            _ldu(0, "The total revenue increased by 15% in FY 2023.", ChunkType.paragraph, page=1),
        ]
        summary = builder._fallback_summary(ldus)
        assert "revenue" in summary.lower()
