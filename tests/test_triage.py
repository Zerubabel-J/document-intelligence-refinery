"""
Unit tests for the Triage Agent.

Tests focus on:
1. Origin type classification logic
2. Extraction cost estimation
3. Domain keyword matching
4. DocumentProfile validation (Pydantic model-level invariants)
5. Profile serialisation round-trip

These tests use no real PDFs — all pdfplumber I/O is mocked so the test
suite runs without any corpus files.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.triage import TriageAgent
from src.models.document_profile import (
    ClassificationSignals,
    DocumentProfile,
    DomainHint,
    ExtractionCost,
    LanguageDetection,
    LayoutComplexity,
    OriginType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_signals(
    avg_chars: float = 1500,
    avg_density: float = 0.003,
    image_ratio: float = 0.05,
    table_count: int = 2,
    figure_count: int = 1,
    font_count: int = 4,
    has_text: bool = True,
    has_forms: bool = False,
) -> ClassificationSignals:
    return ClassificationSignals(
        avg_chars_per_page=avg_chars,
        avg_char_density=avg_density,
        image_area_ratio=image_ratio,
        table_count_estimate=table_count,
        figure_count_estimate=figure_count,
        font_count=font_count,
        has_embedded_text=has_text,
        has_form_fields=has_forms,
        pages_sampled=5,
    )


@pytest.fixture()
def agent(tmp_path) -> TriageAgent:
    """Return a TriageAgent without any rules file (uses defaults)."""
    return TriageAgent(rules_path=str(tmp_path / "nonexistent.yaml"))


# ── origin_type tests ─────────────────────────────────────────────────────────

class TestOriginTypeClassification:

    def test_native_digital_high_text(self, agent):
        signals = _make_signals(avg_chars=1800, image_ratio=0.05)
        result = agent._classify_origin(signals)
        assert result == OriginType.native_digital

    def test_scanned_no_embedded_text(self, agent):
        signals = _make_signals(has_text=False)
        result = agent._classify_origin(signals)
        assert result == OriginType.scanned_image

    def test_scanned_low_chars_high_image(self, agent):
        signals = _make_signals(avg_chars=10, image_ratio=0.80, has_text=True)
        result = agent._classify_origin(signals)
        assert result == OriginType.scanned_image

    def test_form_fillable(self, agent):
        signals = _make_signals(has_forms=True, avg_chars=500)
        result = agent._classify_origin(signals)
        assert result == OriginType.form_fillable

    def test_mixed_digital_with_images(self, agent):
        # High char count but also significant image coverage → mixed
        signals = _make_signals(avg_chars=900, image_ratio=0.40)
        result = agent._classify_origin(signals)
        assert result == OriginType.mixed


# ── layout_complexity tests ───────────────────────────────────────────────────

class TestLayoutComplexityClassification:

    def test_single_column_simple(self, agent):
        signals = _make_signals(table_count=1, figure_count=1, font_count=3)
        result = agent._classify_layout(signals)
        assert result == LayoutComplexity.single_column

    def test_table_heavy(self, agent):
        signals = _make_signals(table_count=8, figure_count=1, font_count=4)
        result = agent._classify_layout(signals)
        assert result == LayoutComplexity.table_heavy

    def test_figure_heavy(self, agent):
        signals = _make_signals(table_count=1, figure_count=6, font_count=4)
        result = agent._classify_layout(signals)
        assert result == LayoutComplexity.figure_heavy

    def test_mixed_tables_and_figures(self, agent):
        signals = _make_signals(table_count=7, figure_count=5, font_count=4)
        result = agent._classify_layout(signals)
        assert result == LayoutComplexity.mixed

    def test_multi_column_many_fonts(self, agent):
        signals = _make_signals(table_count=2, figure_count=2, font_count=8)
        result = agent._classify_layout(signals)
        assert result == LayoutComplexity.multi_column


# ── extraction cost estimation tests ─────────────────────────────────────────

class TestExtractionCostEstimation:

    def test_fast_text_for_simple_digital(self, agent):
        cost = agent._estimate_cost(OriginType.native_digital, LayoutComplexity.single_column)
        assert cost == ExtractionCost.fast_text_sufficient

    def test_vision_for_scanned(self, agent):
        cost = agent._estimate_cost(OriginType.scanned_image, LayoutComplexity.single_column)
        assert cost == ExtractionCost.needs_vision_model

    def test_layout_for_table_heavy(self, agent):
        cost = agent._estimate_cost(OriginType.native_digital, LayoutComplexity.table_heavy)
        assert cost == ExtractionCost.needs_layout_model

    def test_layout_for_multi_column(self, agent):
        cost = agent._estimate_cost(OriginType.native_digital, LayoutComplexity.multi_column)
        assert cost == ExtractionCost.needs_layout_model

    def test_layout_for_mixed_origin(self, agent):
        cost = agent._estimate_cost(OriginType.mixed, LayoutComplexity.single_column)
        assert cost == ExtractionCost.needs_layout_model


# ── domain classification tests ───────────────────────────────────────────────

class TestDomainClassification:

    def test_financial_keywords_detected(self, agent, tmp_path):
        """
        Test domain classification using the filename-based fallback path.
        Naming the file with financial keywords triggers the financial domain.
        """
        pdf = tmp_path / "annual_report_revenue_profit_balance_sheet.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        # Disable pdfplumber so the stem-based fallback activates
        with patch("src.agents.triage._PDFPLUMBER_AVAILABLE", False):
            domain, keywords = agent._classify_domain(pdf)

        assert domain == DomainHint.financial
        assert len(keywords) >= 2

    def test_general_fallback_no_keywords(self, agent, tmp_path):
        pdf = tmp_path / "misc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        with patch("src.agents.triage._PDFPLUMBER_AVAILABLE", False):
            domain, keywords = agent._classify_domain(pdf)
        assert domain == DomainHint.general
        assert keywords == []


# ── DocumentProfile model validation tests ───────────────────────────────────

class TestDocumentProfileValidation:

    def test_valid_profile_constructs(self):
        profile = DocumentProfile(
            doc_id="abc123",
            doc_name="test.pdf",
            doc_path="/tmp/test.pdf",
            page_count=10,
            origin_type=OriginType.native_digital,
            layout_complexity=LayoutComplexity.single_column,
            language=LanguageDetection(language_code="en", confidence=0.95),
            domain_hint=DomainHint.financial,
            estimated_extraction_cost=ExtractionCost.fast_text_sufficient,
            signals=_make_signals(),
        )
        assert profile.doc_id == "abc123"

    def test_scanned_with_fast_text_raises(self):
        """scanned_image + fast_text_sufficient violates invariant → ValidationError."""
        with pytest.raises(Exception):
            DocumentProfile(
                doc_id="bad",
                doc_name="bad.pdf",
                doc_path="/tmp/bad.pdf",
                page_count=5,
                origin_type=OriginType.scanned_image,
                layout_complexity=LayoutComplexity.single_column,
                language=LanguageDetection(language_code="en", confidence=0.9),
                domain_hint=DomainHint.general,
                estimated_extraction_cost=ExtractionCost.fast_text_sufficient,  # invalid!
                signals=_make_signals(has_text=False),
            )

    def test_profile_json_round_trip(self):
        profile = DocumentProfile(
            doc_id="round_trip",
            doc_name="doc.pdf",
            doc_path="/tmp/doc.pdf",
            page_count=20,
            origin_type=OriginType.mixed,
            layout_complexity=LayoutComplexity.table_heavy,
            language=LanguageDetection(language_code="en", confidence=0.88),
            domain_hint=DomainHint.legal,
            estimated_extraction_cost=ExtractionCost.needs_layout_model,
            signals=_make_signals(),
        )
        json_str = profile.model_dump_json()
        restored = DocumentProfile.model_validate_json(json_str)
        assert restored.doc_id == profile.doc_id
        assert restored.origin_type == profile.origin_type
        assert restored.estimated_extraction_cost == profile.estimated_extraction_cost
