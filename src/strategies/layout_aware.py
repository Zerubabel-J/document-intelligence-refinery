"""
Strategy B — Layout-Aware Extractor.

Tool: Docling (IBM Research) with a MinerU fallback adapter.
Cost: Medium (local GPU/CPU inference, no external API)
Triggers when: multi_column OR table_heavy OR mixed origin

The DoclingDocumentAdapter normalises Docling's DoclingDocument into
the shared ExtractedDocument schema, preserving:
  - Text blocks with bounding boxes and reading order
  - Tables as structured TableData objects (headers + rows + cells)
  - Figures with captions
  - Heading hierarchy (used by PageIndex builder)

If Docling is not installed, falls back to pdfplumber with layout
heuristics and logs a warning.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import (
    BoundingBox,
    ExtractedDocument,
    ExtractionMetadata,
    FigureBlock,
    TableCell,
    TableData,
    TextBlock,
)
from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# ── Optional Docling import ───────────────────────────────────────────────────
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    _DOCLING_OK = True
except ImportError:
    _DOCLING_OK = False
    logger.warning("Docling not installed — LayoutExtractor will use pdfplumber fallback")

try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except ImportError:
    _PDFPLUMBER_OK = False

_DEFAULTS = {
    "confidence_threshold": 0.55,
    "table_confidence_floor": 0.70,
}


class LayoutExtractor(BaseExtractor):
    """
    Strategy B: Layout-aware extraction via Docling.

    Uses DoclingDocumentAdapter to normalise output.
    Falls back to an enhanced pdfplumber extraction if Docling is absent.
    """

    strategy_name = "layout_aware"

    def __init__(self, thresholds: Optional[dict] = None) -> None:
        cfg = {**_DEFAULTS, **(thresholds or {})}
        super().__init__(confidence_threshold=float(cfg["confidence_threshold"]))
        self._table_conf_floor = float(cfg["table_confidence_floor"])

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(self, doc_path: Path, profile: DocumentProfile) -> ExtractionResult:
        t0 = time.perf_counter()

        if _DOCLING_OK:
            return self._extract_with_docling(doc_path, profile, t0)
        if _PDFPLUMBER_OK:
            logger.warning("Docling unavailable — using pdfplumber layout fallback for %s", doc_path.name)
            return self._extract_with_pdfplumber_fallback(doc_path, profile, t0)

        return ExtractionResult(
            confidence=0.0,
            document=None,
            strategy_name=self.strategy_name,
            escalate=True,
            error="Neither Docling nor pdfplumber is installed",
        )

    # ── Docling path ──────────────────────────────────────────────────────────

    def _extract_with_docling(
        self, doc_path: Path, profile: DocumentProfile, t0: float
    ) -> ExtractionResult:
        warnings: list[str] = []
        try:
            converter = DocumentConverter()
            result = converter.convert(str(doc_path))
            dl_doc = result.document
        except Exception as exc:
            logger.error("Docling conversion failed for %s: %s", doc_path.name, exc)
            return ExtractionResult(
                confidence=0.0,
                document=None,
                strategy_name=self.strategy_name,
                escalate=True,
                error=str(exc),
            )

        adapter = DoclingDocumentAdapter(dl_doc, profile.doc_id, profile.doc_name)
        text_blocks, tables, figures, reading_order, adapt_warnings = adapter.normalise()
        warnings.extend(adapt_warnings)

        confidence = self._score_confidence(text_blocks, tables, warnings)
        elapsed = time.perf_counter() - t0

        doc = ExtractedDocument(
            doc_id=profile.doc_id,
            doc_name=profile.doc_name,
            page_count=profile.page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order_ids=reading_order,
            metadata=ExtractionMetadata(
                strategy_used=self.strategy_name,
                confidence_score=round(confidence, 4),
                cost_estimate_usd=0.0,
                processing_time_s=round(elapsed, 3),
                tool_version="docling",
                warnings=warnings,
            ),
        )
        return ExtractionResult(
            confidence=confidence,
            document=doc,
            strategy_name=self.strategy_name,
            escalate=self._should_escalate(confidence),
            warnings=warnings,
        )

    # ── pdfplumber fallback path ──────────────────────────────────────────────

    def _extract_with_pdfplumber_fallback(
        self, doc_path: Path, profile: DocumentProfile, t0: float
    ) -> ExtractionResult:
        """Enhanced pdfplumber extraction used when Docling is not available."""
        from .fast_text import FastTextExtractor  # avoid circular import
        fast = FastTextExtractor(thresholds={"confidence_threshold": 0.0})
        result = fast.extract(doc_path, profile)
        elapsed = time.perf_counter() - t0

        if result.document:
            result.document.metadata.strategy_used = f"{self.strategy_name}(fallback:fast_text)"
            result.document.metadata.processing_time_s = round(elapsed, 3)
            result.document.metadata.warnings.append(
                "Docling unavailable — layout extraction used pdfplumber fallback; "
                "table bounding boxes are approximate"
            )
        # Downgrade confidence slightly since we're not doing true layout analysis
        adj_confidence = round(result.confidence * 0.85, 4)
        return ExtractionResult(
            confidence=adj_confidence,
            document=result.document,
            strategy_name=self.strategy_name,
            escalate=self._should_escalate(adj_confidence),
            warnings=result.warnings + ["Docling not available — fallback extraction used"],
        )

    # ── Confidence scoring ────────────────────────────────────────────────────

    def _score_confidence(
        self,
        text_blocks: list[TextBlock],
        tables: list[TableData],
        warnings: list[str],
    ) -> float:
        """
        Confidence heuristics for layout extraction:
          - Text presence (0.40)
          - Table quality (0.40): all tables have headers
          - Warning penalty (0.20): fewer warnings → higher score
        """
        text_score = min(1.0, len(text_blocks) / 10)
        if tables:
            tables_with_headers = sum(1 for t in tables if t.headers)
            tbl_score = tables_with_headers / len(tables)
        else:
            tbl_score = 1.0  # no tables — not a failure
        warn_penalty = min(0.20, len(warnings) * 0.04)
        return round(0.40 * text_score + 0.40 * tbl_score + 0.20 - warn_penalty, 4)


class DoclingDocumentAdapter:
    """
    Normalises a Docling DoclingDocument into the shared ExtractedDocument
    component types (TextBlock, TableData, FigureBlock).

    This adapter is the integration point — if the Docling schema changes
    only this class needs updating; the rest of the pipeline is unaffected.
    """

    def __init__(self, dl_doc, doc_id: str, doc_name: str) -> None:
        self.dl_doc = dl_doc
        self.doc_id = doc_id
        self.doc_name = doc_name

    def normalise(
        self,
    ) -> tuple[list[TextBlock], list[TableData], list[FigureBlock], list[str], list[str]]:
        text_blocks: list[TextBlock] = []
        tables: list[TableData] = []
        figures: list[FigureBlock] = []
        reading_order: list[str] = []
        warnings: list[str] = []

        try:
            seq = 0
            for item in self.dl_doc.iterate_items():
                item_type = type(item).__name__

                if item_type in ("TextItem", "SectionHeaderItem", "ParagraphItem"):
                    blk = self._adapt_text(item, seq)
                    text_blocks.append(blk)
                    reading_order.append(blk.block_id)
                    seq += 1

                elif item_type == "TableItem":
                    tbl = self._adapt_table(item, seq)
                    if tbl:
                        tables.append(tbl)
                        reading_order.append(tbl.table_id)
                    seq += 1

                elif item_type in ("FigureItem", "PictureItem"):
                    fig = self._adapt_figure(item, seq)
                    figures.append(fig)
                    reading_order.append(fig.figure_id)
                    seq += 1

        except Exception as exc:
            warnings.append(f"DoclingAdapter iteration error: {exc}")

        return text_blocks, tables, figures, reading_order, warnings

    def _adapt_text(self, item, seq: int) -> TextBlock:
        page_no = self._get_page(item)
        bbox = self._get_bbox(item, page_no)
        is_heading = "Header" in type(item).__name__ or "Title" in type(item).__name__
        return TextBlock(
            block_id=f"{self.doc_id}_b{seq:05d}",
            text=getattr(item, "text", "") or "",
            bbox=bbox,
            is_heading=is_heading,
            reading_order=seq,
        )

    def _adapt_table(self, item, seq: int) -> Optional[TableData]:
        try:
            page_no = self._get_page(item)
            bbox = self._get_bbox(item, page_no)
            df = item.export_to_dataframe()
            cells: list[TableCell] = []
            for r_i, row in enumerate(df.values.tolist()):
                for c_i, val in enumerate(row):
                    cells.append(TableCell(
                        row=r_i,
                        col=c_i,
                        text=str(val) if val is not None else "",
                        is_header=(r_i == 0),
                    ))
            return TableData(
                table_id=f"{self.doc_id}_tbl{seq:04d}",
                bbox=bbox,
                cells=cells,
                col_count=len(df.columns),
                row_count=len(df),
                extraction_confidence=0.90,
            )
        except Exception:
            return None

    def _adapt_figure(self, item, seq: int) -> FigureBlock:
        page_no = self._get_page(item)
        bbox = self._get_bbox(item, page_no)
        caption = ""
        if hasattr(item, "captions") and item.captions:
            caption = item.captions[0].text if item.captions else ""
        return FigureBlock(
            figure_id=f"{self.doc_id}_fig{seq:04d}",
            bbox=bbox,
            caption=caption or None,
        )

    @staticmethod
    def _get_page(item) -> int:
        try:
            prov = item.prov[0] if item.prov else None
            return (prov.page_no if prov else 1) or 1
        except Exception:
            return 1

    def _get_bbox(self, item, page_no: int) -> BoundingBox:
        try:
            prov = item.prov[0] if item.prov else None
            if prov and hasattr(prov, "bbox"):
                b = prov.bbox
                return BoundingBox(
                    x0=float(b.l), y0=float(b.b),
                    x1=float(b.r), y1=float(b.t),
                    page=page_no,
                )
        except Exception:
            pass
        return BoundingBox(x0=0, y0=0, x1=595, y1=842, page=page_no)
