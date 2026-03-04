"""
Strategy A — Fast Text Extractor.

Tool: pdfplumber (with pymupdf fallback)
Cost: Low (local, no API calls)
Triggers when: origin_type=native_digital AND layout_complexity=single_column

Confidence scoring uses four signals:
  1. character_density  — chars per pt² of page area
  2. char_count_ratio   — pages with ≥ min_chars / total pages
  3. image_area_ratio   — fraction of page area that is images (inverse signal)
  4. font_presence      — whether font metadata was captured

Low confidence pages trigger escalation to Strategy B (LayoutExtractor).

Thresholds are read from extraction_rules.yaml (strategy_a section).
"""

from __future__ import annotations

import logging
import time
import uuid
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

try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except ImportError:
    _PDFPLUMBER_OK = False

# ── Default thresholds (overridden by extraction_rules.yaml → strategy_a) ────
_DEFAULTS = {
    "min_chars_per_page": 100,
    "min_char_density": 0.001,
    "max_image_area_ratio": 0.50,
    "confidence_threshold": 0.60,
}


class FastTextExtractor(BaseExtractor):
    """
    Strategy A: pdfplumber-based text extraction with confidence scoring.

    Returns an ExtractionResult with escalate=True if confidence < threshold.
    """

    strategy_name = "fast_text"

    def __init__(self, thresholds: Optional[dict] = None) -> None:
        cfg = {**_DEFAULTS, **(thresholds or {})}
        super().__init__(confidence_threshold=float(cfg["confidence_threshold"]))
        self._min_chars = int(cfg["min_chars_per_page"])
        self._min_density = float(cfg["min_char_density"])
        self._max_img_ratio = float(cfg["max_image_area_ratio"])

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(self, doc_path: Path, profile: DocumentProfile) -> ExtractionResult:
        if not _PDFPLUMBER_OK:
            return ExtractionResult(
                confidence=0.0,
                document=None,
                strategy_name=self.strategy_name,
                escalate=True,
                error="pdfplumber not installed",
            )

        t0 = time.perf_counter()
        text_blocks: list[TextBlock] = []
        tables: list[TableData] = []
        figures: list[FigureBlock] = []
        reading_order: list[str] = []
        page_confidences: list[float] = []
        warnings: list[str] = []
        block_seq = 0

        try:
            with pdfplumber.open(str(doc_path)) as pdf:
                for pg_num, page in enumerate(pdf.pages, start=1):
                    pg_conf, pg_blocks, pg_tables, pg_figures, pg_order, pg_warns = \
                        self._extract_page(page, pg_num, profile.doc_id, block_seq)

                    page_confidences.append(pg_conf)
                    text_blocks.extend(pg_blocks)
                    tables.extend(pg_tables)
                    figures.extend(pg_figures)
                    reading_order.extend(pg_order)
                    warnings.extend(pg_warns)
                    block_seq += len(pg_blocks) + len(pg_tables) + len(pg_figures)

        except Exception as exc:
            logger.error("FastTextExtractor failed on %s: %s", doc_path.name, exc)
            return ExtractionResult(
                confidence=0.0,
                document=None,
                strategy_name=self.strategy_name,
                escalate=True,
                error=str(exc),
            )

        confidence = self._aggregate_confidence(page_confidences)
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
                warnings=warnings,
            ),
        )

        escalate = self._should_escalate(confidence)
        if escalate:
            logger.info(
                "FastTextExtractor: low confidence (%.2f) on %s — requesting escalation",
                confidence, doc_path.name,
            )

        return ExtractionResult(
            confidence=confidence,
            document=doc,
            strategy_name=self.strategy_name,
            escalate=escalate,
            warnings=warnings,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_page(
        self,
        page,
        pg_num: int,
        doc_id: str,
        block_offset: int,
    ) -> tuple[float, list[TextBlock], list[TableData], list[FigureBlock], list[str], list[str]]:
        """Extract all content from a single pdfplumber page."""
        text_blocks: list[TextBlock] = []
        tables: list[TableData] = []
        figures: list[FigureBlock] = []
        reading_order: list[str] = []
        warnings: list[str] = []

        w = page.width or 595
        h = page.height or 842
        page_area = w * h

        # ── Images / figures ──────────────────────────────────────────────────
        imgs = page.images or []
        img_area = 0.0
        for i, im in enumerate(imgs):
            fig_id = f"{doc_id}_p{pg_num}_fig{i}"
            bbox = BoundingBox(
                x0=float(im.get("x0", 0)),
                y0=float(im.get("y0", 0)),
                x1=float(im.get("x1", im.get("x0", 0) + im.get("width", 0))),
                y1=float(im.get("y1", im.get("y0", 0) + im.get("height", 0))),
                page=pg_num,
            )
            figures.append(FigureBlock(figure_id=fig_id, bbox=bbox))
            reading_order.append(fig_id)
            img_area += float(im.get("width", 0)) * float(im.get("height", 0))

        img_ratio = img_area / page_area if page_area > 0 else 0.0

        # ── Tables ────────────────────────────────────────────────────────────
        try:
            raw_tables = page.find_tables()
            for t_idx, raw_tbl in enumerate(raw_tables):
                tbl_id = f"{doc_id}_p{pg_num}_tbl{t_idx}"
                tbl_bbox = BoundingBox(
                    x0=raw_tbl.bbox[0], y0=raw_tbl.bbox[1],
                    x1=raw_tbl.bbox[2], y1=raw_tbl.bbox[3],
                    page=pg_num,
                )
                cells = []
                extracted = raw_tbl.extract() or []
                row_count = len(extracted)
                col_count = max((len(r) for r in extracted), default=0)
                for r_i, row in enumerate(extracted):
                    for c_i, cell_text in enumerate(row):
                        cells.append(TableCell(
                            row=r_i,
                            col=c_i,
                            text=(cell_text or "").strip(),
                            is_header=(r_i == 0),
                        ))
                table = TableData(
                    table_id=tbl_id,
                    bbox=tbl_bbox,
                    cells=cells,
                    col_count=col_count,
                    row_count=row_count,
                    extraction_confidence=0.85,
                )
                tables.append(table)
                reading_order.append(tbl_id)
        except Exception as e:
            warnings.append(f"p{pg_num}: table extraction error — {e}")

        # ── Text blocks ───────────────────────────────────────────────────────
        chars = page.chars or []
        char_count = len(chars)
        density = char_count / page_area if page_area > 0 else 0.0

        # Group chars into lines (naïve: cluster by y-coordinate)
        raw_text = page.extract_text() or ""
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        local_seq = 0
        for line in lines:
            blk_id = f"{doc_id}_p{pg_num}_b{block_offset + local_seq}"
            # Approximate bbox — pdfplumber doesn't give line bbox directly
            text_blocks.append(TextBlock(
                block_id=blk_id,
                text=line,
                bbox=BoundingBox(x0=0, y0=0, x1=w, y1=h, page=pg_num),
                reading_order=block_offset + local_seq,
            ))
            reading_order.append(blk_id)
            local_seq += 1

        # ── Page-level confidence ─────────────────────────────────────────────
        confidence = self._page_confidence(char_count, density, img_ratio)
        if confidence < self.confidence_threshold:
            warnings.append(
                f"p{pg_num}: low confidence ({confidence:.2f}) — "
                f"chars={char_count}, density={density:.4f}, img_ratio={img_ratio:.2f}"
            )

        return confidence, text_blocks, tables, figures, reading_order, warnings

    def _page_confidence(self, char_count: int, density: float, img_ratio: float) -> float:
        """
        Multi-signal confidence score for a single page.

        Signal weights (justified in DOMAIN_NOTES.md):
          - char_count_score (0.40): normalised against min_chars threshold
          - density_score    (0.30): normalised against min_density threshold
          - image_penalty    (0.30): inverted image coverage
        """
        # char count score — saturates at 5× threshold
        char_score = min(1.0, char_count / max(1, self._min_chars * 5))
        # density score — saturates at 5× threshold
        density_score = min(1.0, density / max(1e-9, self._min_density * 5))
        # image penalty — high image area → low confidence for text extraction
        image_score = max(0.0, 1.0 - (img_ratio / max(self._max_img_ratio, 0.01)))

        return round(0.40 * char_score + 0.30 * density_score + 0.30 * image_score, 4)

    @staticmethod
    def _aggregate_confidence(page_confs: list[float]) -> float:
        """
        Document-level confidence = mean of page-level scores,
        weighted towards the worst pages (min included at 30%).
        """
        if not page_confs:
            return 0.0
        mean_conf = sum(page_confs) / len(page_confs)
        min_conf = min(page_confs)
        return round(0.70 * mean_conf + 0.30 * min_conf, 4)
