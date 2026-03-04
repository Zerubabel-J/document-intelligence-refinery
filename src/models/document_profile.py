"""
DocumentProfile — the typed output of the Triage Agent.

Every downstream stage reads this profile to decide which extraction
strategy to apply, which prompt template to use, and how much compute
to budget.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class OriginType(str, Enum):
    """How the PDF was created — determines whether a character stream exists."""
    native_digital = "native_digital"   # Created by word-processor / typesetter
    scanned_image = "scanned_image"     # Pure raster scan; no character stream
    mixed = "mixed"                     # Some pages digital, some scanned
    form_fillable = "form_fillable"     # AcroForm / XFA with interactive fields


class LayoutComplexity(str, Enum):
    """Structural complexity of the page layout."""
    single_column = "single_column"
    multi_column = "multi_column"
    table_heavy = "table_heavy"
    figure_heavy = "figure_heavy"
    mixed = "mixed"


class DomainHint(str, Enum):
    """
    Broad content domain — selects prompt strategy and fact-extraction rules.
    Kept as a hint because classification is probabilistic.
    """
    financial = "financial"
    legal = "legal"
    technical = "technical"
    medical = "medical"
    general = "general"


class ExtractionCost(str, Enum):
    """
    Estimated extraction cost tier selected by the Triage Agent.
    Maps directly to the three extraction strategies.
    """
    fast_text_sufficient = "fast_text_sufficient"   # Strategy A — pdfplumber / pymupdf
    needs_layout_model = "needs_layout_model"        # Strategy B — MinerU / Docling
    needs_vision_model = "needs_vision_model"        # Strategy C — VLM (Gemini / GPT-4o)


class LanguageDetection(BaseModel):
    """Language classification result with confidence."""
    language_code: str = Field(description="ISO 639-1 language code, e.g. 'en', 'am'")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier confidence in [0, 1]")
    secondary_language: Optional[str] = Field(
        default=None,
        description="Second language present in the document (e.g. bilingual report)",
    )


class ClassificationSignals(BaseModel):
    """
    Raw heuristic signals captured during triage.
    Stored alongside the profile for auditability and threshold tuning.
    """
    avg_chars_per_page: float = Field(description="Mean character count across sampled pages")
    avg_char_density: float = Field(
        description="Characters per point² of page area (sampled pages)"
    )
    image_area_ratio: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of total page area occupied by embedded images",
    )
    table_count_estimate: int = Field(description="Heuristic count of table-like regions")
    figure_count_estimate: int = Field(description="Heuristic count of figure bounding boxes")
    font_count: int = Field(description="Number of distinct font names encountered")
    has_embedded_text: bool = Field(description="True if any page has a character stream")
    has_form_fields: bool = Field(description="True if AcroForm / XFA fields were detected")
    pages_sampled: int = Field(description="Number of pages analysed during triage")


class DocumentProfile(BaseModel):
    """
    The authoritative classification record produced by the Triage Agent.

    Stored at  .refinery/profiles/{doc_id}.json  and read by every
    downstream stage.  All fields are immutable after creation — if a
    document is re-triaged a new profile with a fresh doc_id is created.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    doc_id: str = Field(description="SHA-256 hex of the raw file bytes (first 16 chars used as prefix)")
    doc_name: str = Field(description="Original filename")
    doc_path: str = Field(description="Absolute or project-relative path to the source file")
    page_count: int = Field(ge=1, description="Total number of pages in the document")

    # ── Classification dimensions ─────────────────────────────────────────────
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language: LanguageDetection
    domain_hint: DomainHint
    estimated_extraction_cost: ExtractionCost

    # ── Supporting evidence ───────────────────────────────────────────────────
    signals: ClassificationSignals = Field(
        description="Raw heuristic signals used to derive the above classifications"
    )
    domain_keywords_matched: list[str] = Field(
        default_factory=list,
        description="Keywords that triggered the domain_hint classification",
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    triage_version: str = Field(default="1.0.0", description="Triage Agent version that produced this profile")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = Field(default=None, description="Free-text notes added during triage")

    @model_validator(mode="after")
    def validate_cost_vs_origin(self) -> "DocumentProfile":
        """
        Enforce consistency: scanned documents cannot be served by fast text extraction
        because they have no character stream.
        """
        if (
            self.origin_type == OriginType.scanned_image
            and self.estimated_extraction_cost == ExtractionCost.fast_text_sufficient
        ):
            raise ValueError(
                "scanned_image origin cannot have fast_text_sufficient cost — "
                "scanned pages have no character stream."
            )
        return self

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}
