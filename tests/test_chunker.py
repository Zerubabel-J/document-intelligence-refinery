"""
Unit tests for the Semantic Chunking Engine.

Tests cover:
1. Five chunking rules enforcement
2. ChunkValidator violation detection
3. Token-based splitting
4. Cross-reference resolution
"""

from __future__ import annotations

import pytest

from src.agents.chunker import ChunkingEngine, ChunkValidator
from src.models.extracted_document import (
    BoundingBox, ExtractedDocument, ExtractionMetadata,
    FigureBlock, TableCell, TableData, TextBlock,
)
from src.models.ldu import LDU, ChunkType


@pytest.fixture()
def engine():
    return ChunkingEngine(rules_path="nonexistent.yaml")


def _bbox(page: int = 1) -> BoundingBox:
    return BoundingBox(x0=0, y0=0, x1=595, y1=842, page=page)


def _make_doc(
    text_blocks: list[TextBlock] | None = None,
    tables: list[TableData] | None = None,
    figures: list[FigureBlock] | None = None,
) -> ExtractedDocument:
    blocks = text_blocks or []
    tbls = tables or []
    figs = figures or []
    order = [b.block_id for b in blocks] + [t.table_id for t in tbls] + [f.figure_id for f in figs]
    return ExtractedDocument(
        doc_id="test_doc",
        doc_name="test.pdf",
        page_count=5,
        text_blocks=blocks,
        tables=tbls,
        figures=figs,
        reading_order_ids=order,
        metadata=ExtractionMetadata(
            strategy_used="fast_text",
            confidence_score=0.90,
            cost_estimate_usd=0.0,
            processing_time_s=1.0,
        ),
    )


class TestChunkingRules:

    def test_rule1_table_preserves_headers(self, engine):
        """Rule 1: table cells never split from header row."""
        table = TableData(
            table_id="tbl1",
            bbox=_bbox(),
            cells=[
                TableCell(row=0, col=0, text="Revenue", is_header=True),
                TableCell(row=0, col=1, text="Q1", is_header=True),
                TableCell(row=1, col=0, text="Total"),
                TableCell(row=1, col=1, text="1,234,567"),
            ],
            col_count=2,
            row_count=2,
        )
        doc = _make_doc(tables=[table])
        ldus = engine.chunk(doc)
        table_ldus = [l for l in ldus if l.chunk_type == ChunkType.table]
        assert len(table_ldus) == 1
        assert "Revenue" in table_ldus[0].content
        assert "1,234,567" in table_ldus[0].content

    def test_rule2_figure_caption_as_metadata(self, engine):
        """Rule 2: figure caption stored as metadata of parent figure chunk."""
        figure = FigureBlock(
            figure_id="fig1",
            bbox=_bbox(),
            caption="Figure 1: Revenue growth trend",
            figure_type="chart",
        )
        doc = _make_doc(figures=[figure])
        ldus = engine.chunk(doc)
        fig_ldus = [l for l in ldus if l.chunk_type == ChunkType.figure]
        assert len(fig_ldus) == 1
        assert "Caption: Figure 1: Revenue growth trend" in fig_ldus[0].content

    def test_rule3_list_kept_as_single_ldu(self, engine):
        """Rule 3: numbered list kept as single LDU."""
        list_text = "1. First item\n2. Second item\n3. Third item"
        block = TextBlock(
            block_id="b1", text=list_text, bbox=_bbox(), reading_order=0,
        )
        doc = _make_doc(text_blocks=[block])
        ldus = engine.chunk(doc)
        list_ldus = [l for l in ldus if l.chunk_type == ChunkType.list_block]
        assert len(list_ldus) == 1
        assert "First item" in list_ldus[0].content
        assert "Third item" in list_ldus[0].content

    def test_rule4_section_header_propagated(self, engine):
        """Rule 4: section headers propagated as parent_section."""
        blocks = [
            TextBlock(block_id="h1", text="Financial Overview", bbox=_bbox(),
                      is_heading=True, heading_level=1, reading_order=0),
            TextBlock(block_id="b1", text="This section covers the annual financial results.",
                      bbox=_bbox(), reading_order=1),
            TextBlock(block_id="b2", text="Revenue increased by 15% year over year.",
                      bbox=_bbox(), reading_order=2),
        ]
        doc = _make_doc(text_blocks=blocks)
        ldus = engine.chunk(doc)
        content_ldus = [l for l in ldus if l.chunk_type == ChunkType.paragraph]
        for ldu in content_ldus:
            assert ldu.parent_section == "Financial Overview"

    def test_rule5_cross_reference_resolved(self, engine):
        """Rule 5: cross-references resolved to chunk relationships."""
        blocks = [
            TextBlock(block_id="b1", text="As shown in Table 1, revenue grew.",
                      bbox=_bbox(), reading_order=0),
        ]
        table = TableData(
            table_id="tbl1", bbox=_bbox(),
            cells=[
                TableCell(row=0, col=0, text="Metric", is_header=True),
                TableCell(row=1, col=0, text="Revenue"),
            ],
            col_count=1, row_count=2,
        )
        doc = _make_doc(text_blocks=blocks, tables=[table])
        ldus = engine.chunk(doc)
        xref_ldus = [l for l in ldus if any(r.relation_type == "cross_ref" for r in l.relationships)]
        assert len(xref_ldus) >= 1


class TestChunkValidator:

    def test_valid_paragraph_passes(self):
        ldu = LDU(
            chunk_id="test_000001", doc_id="test", sequence=1,
            content="A valid paragraph.", chunk_type=ChunkType.paragraph,
            token_count=4, page_refs=[1], parent_section="Introduction",
            content_hash="abc",
        )
        violations = ChunkValidator.validate(ldu, {"has_sections": True})
        assert violations == []

    def test_missing_parent_section_flagged(self):
        ldu = LDU(
            chunk_id="test_000002", doc_id="test", sequence=2,
            content="Missing section.", chunk_type=ChunkType.paragraph,
            token_count=3, page_refs=[1], parent_section=None,
            content_hash="def",
        )
        violations = ChunkValidator.validate(ldu, {"has_sections": True})
        assert any("Rule 4" in v for v in violations)


class TestContentHash:

    def test_content_hash_auto_computed(self, engine):
        block = TextBlock(
            block_id="b1", text="Hash test content.",
            bbox=_bbox(), reading_order=0,
        )
        doc = _make_doc(text_blocks=[block])
        ldus = engine.chunk(doc)
        assert len(ldus) >= 1
        assert ldus[0].content_hash
        assert len(ldus[0].content_hash) == 64  # SHA-256 hex length
