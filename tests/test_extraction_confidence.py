"""
Unit tests for extraction confidence scoring.

Tests cover:
1. FastTextExtractor page-level confidence calculation
2. FastTextExtractor document-level (aggregate) confidence
3. ExtractionResult escalation logic
4. LayoutExtractor confidence scoring
"""

from __future__ import annotations

import pytest

from src.strategies.fast_text import FastTextExtractor
from src.strategies.base import ExtractionResult
from src.strategies.layout_aware import LayoutExtractor
from src.models.extracted_document import TextBlock, TableData, TableCell, BoundingBox


# ── FastTextExtractor confidence tests ───────────────────────────────────────

class TestFastTextConfidence:
    """Test the multi-signal confidence scoring in Strategy A."""

    @pytest.fixture()
    def extractor(self):
        return FastTextExtractor(thresholds={
            "min_chars_per_page": 100,
            "min_char_density": 0.001,
            "max_image_area_ratio": 0.50,
            "confidence_threshold": 0.60,
        })

    def test_high_confidence_dense_text(self, extractor):
        # 2000 chars, density 0.004, image ratio 0.02 → all signals high
        conf = extractor._page_confidence(2000, 0.004, 0.02)
        assert conf >= 0.80, f"Expected ≥ 0.80, got {conf}"

    def test_low_confidence_scanned_like_page(self, extractor):
        # 5 chars, density 0.00001, image ratio 0.85 → near-zero confidence
        conf = extractor._page_confidence(5, 0.00001, 0.85)
        assert conf < 0.30, f"Expected < 0.30, got {conf}"

    def test_medium_confidence_partial_scan(self, extractor):
        # 200 chars, density 0.0004, image ratio 0.45 → borderline
        conf = extractor._page_confidence(200, 0.0004, 0.45)
        assert 0.20 < conf < 0.80

    def test_zero_chars_zero_density(self, extractor):
        conf = extractor._page_confidence(0, 0.0, 0.0)
        assert conf >= 0.0
        assert conf <= 1.0

    def test_confidence_below_threshold_requests_escalation(self, extractor):
        low_conf = 0.40
        assert extractor._should_escalate(low_conf) is True

    def test_confidence_above_threshold_no_escalation(self, extractor):
        high_conf = 0.80
        assert extractor._should_escalate(high_conf) is False

    def test_aggregate_confidence_penalises_min(self, extractor):
        """Document confidence weights min page (30%) — worst page drags score down."""
        page_confs = [0.90, 0.90, 0.90, 0.10]  # one bad page
        doc_conf = extractor._aggregate_confidence(page_confs)
        mean_only = sum(page_confs) / len(page_confs)  # 0.70
        assert doc_conf < mean_only, "min weighting should pull confidence below mean"

    def test_aggregate_confidence_empty_list(self, extractor):
        assert extractor._aggregate_confidence([]) == 0.0

    def test_aggregate_confidence_single_page(self, extractor):
        conf = extractor._aggregate_confidence([0.75])
        assert conf == pytest.approx(0.75, abs=0.01)


# ── LayoutExtractor confidence tests ─────────────────────────────────────────

class TestLayoutExtractorConfidence:

    @pytest.fixture()
    def extractor(self):
        return LayoutExtractor()

    def _make_text_block(self, n: int) -> list[TextBlock]:
        return [
            TextBlock(
                block_id=f"b{i}",
                text=f"paragraph {i}",
                bbox=BoundingBox(x0=0, y0=0, x1=595, y1=842, page=1),
                reading_order=i,
            )
            for i in range(n)
        ]

    def _make_table(self, with_headers: bool) -> TableData:
        cells = [
            TableCell(row=0, col=0, text="Header A", is_header=True),
            TableCell(row=0, col=1, text="Header B", is_header=True),
            TableCell(row=1, col=0, text="Value 1"),
            TableCell(row=1, col=1, text="Value 2"),
        ]
        if not with_headers:
            for c in cells:
                c.is_header = False
        return TableData(
            table_id="tbl0",
            bbox=BoundingBox(x0=0, y0=0, x1=595, y1=200, page=1),
            cells=cells,
            col_count=2,
            row_count=2,
        )

    def test_high_confidence_text_and_tables(self, extractor):
        blocks = self._make_text_block(15)
        tables = [self._make_table(with_headers=True)]
        conf = extractor._score_confidence(blocks, tables, [])
        assert conf >= 0.70

    def test_lower_confidence_tables_without_headers(self, extractor):
        blocks = self._make_text_block(15)
        tables = [self._make_table(with_headers=False)]
        conf_good = extractor._score_confidence(blocks, [self._make_table(with_headers=True)], [])
        conf_bad = extractor._score_confidence(blocks, tables, [])
        assert conf_bad < conf_good

    def test_warnings_reduce_confidence(self, extractor):
        blocks = self._make_text_block(15)
        conf_no_warns = extractor._score_confidence(blocks, [], [])
        conf_with_warns = extractor._score_confidence(blocks, [], ["w"] * 5)
        assert conf_with_warns < conf_no_warns


# ── ExtractionResult behaviour ────────────────────────────────────────────────

class TestExtractionResult:

    def test_success_true_when_document_present(self):
        from src.models.extracted_document import ExtractedDocument, ExtractionMetadata
        doc = ExtractedDocument(
            doc_id="d1",
            doc_name="test.pdf",
            page_count=1,
            metadata=ExtractionMetadata(
                strategy_used="fast_text",
                confidence_score=0.85,
                cost_estimate_usd=0.0,
                processing_time_s=0.5,
            ),
        )
        result = ExtractionResult(confidence=0.85, document=doc, strategy_name="fast_text")
        assert result.success is True

    def test_success_false_when_document_none(self):
        result = ExtractionResult(
            confidence=0.0,
            document=None,
            strategy_name="fast_text",
            error="pdfplumber failed",
        )
        assert result.success is False

    def test_escalate_flag_propagates(self):
        result = ExtractionResult(
            confidence=0.40,
            document=None,
            strategy_name="fast_text",
            escalate=True,
        )
        assert result.escalate is True
