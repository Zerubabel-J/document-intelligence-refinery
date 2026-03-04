"""
ExtractedDocument — the normalised intermediate representation.

All three extraction strategies (FastText, Layout, Vision) MUST produce
an ExtractedDocument.  This shared schema is the contract that allows the
downstream ChunkingEngine and PageIndex builder to be strategy-agnostic.

Design note: loosely inspired by Docling's DoclingDocument but kept
lightweight so any strategy can populate it without importing Docling.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """
    PDF coordinate bounding box.

    pdfplumber uses the PDF coordinate system:
      origin at bottom-left, x increases right, y increases up.
    Coordinates are in PDF points (1 pt = 1/72 inch).
    """
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = Field(ge=1, description="1-indexed page number")

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "page": self.page}


class TextBlock(BaseModel):
    """A contiguous block of text with spatial metadata."""
    block_id: str = Field(description="Unique ID within this document, e.g. 'p3_b7'")
    text: str
    bbox: BoundingBox
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    is_heading: bool = False
    heading_level: Optional[int] = Field(default=None, ge=1, le=6)
    reading_order: int = Field(description="Zero-based position in the reconstructed reading order")


class TableCell(BaseModel):
    """A single cell within a table, with its row/column address."""
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    text: str
    is_header: bool = False
    colspan: int = Field(default=1, ge=1)
    rowspan: int = Field(default=1, ge=1)


class TableData(BaseModel):
    """
    A table extracted as a structured object.

    The `cells` list is the ground truth; `headers` and `rows` are
    convenience views derived from cells where is_header=True/False.
    """
    table_id: str
    bbox: BoundingBox
    caption: Optional[str] = None
    cells: list[TableCell]
    col_count: int
    row_count: int
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    @property
    def headers(self) -> list[str]:
        return [c.text for c in self.cells if c.is_header and c.row == 0]

    @property
    def rows(self) -> list[list[str]]:
        if not self.cells:
            return []
        max_row = max(c.row for c in self.cells)
        result = []
        for r in range(1, max_row + 1):
            row_cells = sorted([c for c in self.cells if c.row == r], key=lambda c: c.col)
            result.append([c.text for c in row_cells])
        return result


class FigureBlock(BaseModel):
    """A figure / image region with optional extracted caption."""
    figure_id: str
    bbox: BoundingBox
    caption: Optional[str] = None
    figure_type: str = Field(default="unknown", description="chart | diagram | photo | equation | unknown")
    alt_text: Optional[str] = Field(
        default=None,
        description="LLM-generated description when vision extraction is used",
    )


class ExtractionMetadata(BaseModel):
    """Provenance metadata attached to every ExtractedDocument."""
    strategy_used: str = Field(description="fast_text | layout_aware | vision_augmented")
    confidence_score: float = Field(ge=0.0, le=1.0)
    cost_estimate_usd: float = Field(ge=0.0, description="Estimated API cost in USD (0 for local strategies)")
    processing_time_s: float = Field(ge=0.0)
    tool_version: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class ExtractedDocument(BaseModel):
    """
    Normalised extraction result — the contract between all extraction
    strategies and the downstream ChunkingEngine.

    Every strategy adapter MUST produce a valid ExtractedDocument.
    """
    doc_id: str
    doc_name: str
    page_count: int

    text_blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    figures: list[FigureBlock] = Field(default_factory=list)

    # Reading order is encoded in TextBlock.reading_order; this list
    # gives the global sequence across all content types.
    reading_order_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of block_id / table_id / figure_id",
    )

    metadata: ExtractionMetadata

    class Config:
        arbitrary_types_allowed = True
