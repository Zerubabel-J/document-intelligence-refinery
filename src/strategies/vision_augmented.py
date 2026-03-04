"""
Strategy C — Vision-Augmented Extractor.

Tool: Multimodal LLM via OpenRouter (Gemini Flash / GPT-4o-mini)
Cost: High (per-page API calls, ~$0.0005–$0.002 / page)
Triggers when:
  - origin_type = scanned_image
  - Strategy A or B confidence < escalation threshold
  - Handwriting detected

Budget guard:
  - Configurable per-document USD cap (default $0.10)
  - Token spend logged per page
  - Processing stops gracefully if budget is exceeded

Structured extraction prompt:
  Each page image is sent with a structured JSON extraction prompt that
  requests: paragraphs (with bbox estimates), tables (headers + rows),
  figures (description), and page-level metadata.

API client: httpx (async-capable, no SDK dependency)
"""

from __future__ import annotations

import base64
import json
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

try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False

try:
    import fitz  # PyMuPDF — for rasterising PDF pages to images
    _FITZ_OK = True
except ImportError:
    _FITZ_OK = False

_DEFAULTS = {
    "confidence_threshold": 0.50,
    "budget_cap_usd": 0.10,
    "model": "google/gemini-flash-1.5",
    "openrouter_api_key_env": "OPENROUTER_API_KEY",
    "cost_per_1k_input_tokens": 0.00035,   # Gemini Flash pricing (approx.)
    "cost_per_1k_output_tokens": 0.00105,
    "dpi": 150,
    "max_pages_per_doc": 50,
}

_EXTRACTION_PROMPT = """\
You are a document intelligence extraction engine.

Extract ALL content from this document page and return it as a JSON object
with the following schema (no markdown fences — raw JSON only):

{
  "page_number": <int>,
  "paragraphs": [
    {
      "text": "<paragraph text>",
      "is_heading": <bool>,
      "heading_level": <1-6 or null>,
      "bbox_estimate": {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
    }
  ],
  "tables": [
    {
      "caption": "<caption or null>",
      "headers": ["col1", "col2", ...],
      "rows": [["v1", "v2", ...], ...],
      "bbox_estimate": {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
    }
  ],
  "figures": [
    {
      "description": "<what the figure shows>",
      "caption": "<caption text or null>",
      "bbox_estimate": {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
    }
  ],
  "has_handwriting": <bool>,
  "language_hint": "<ISO 639-1 code>"
}

Important rules:
- Extract ALL text — do not summarise.
- For tables, reproduce exact cell values.
- bbox_estimate coordinates are percentages of page dimensions (0–100).
- If a field is absent, use an empty list or null.
"""


class VisionExtractor(BaseExtractor):
    """
    Strategy C: Vision-augmented extraction using a multimodal LLM.

    Each page is rasterised to a PNG and sent with a structured prompt.
    A budget_guard stops processing if the per-document cost cap is hit.
    """

    strategy_name = "vision_augmented"

    def __init__(self, thresholds: Optional[dict] = None, api_key: Optional[str] = None) -> None:
        cfg = {**_DEFAULTS, **(thresholds or {})}
        super().__init__(confidence_threshold=float(cfg["confidence_threshold"]))
        self._budget_cap = float(cfg["budget_cap_usd"])
        self._model = str(cfg["model"])
        self._cost_in = float(cfg["cost_per_1k_input_tokens"])
        self._cost_out = float(cfg["cost_per_1k_output_tokens"])
        self._dpi = int(cfg["dpi"])
        self._max_pages = int(cfg["max_pages_per_doc"])

        # Resolve API key
        import os
        env_key = str(cfg["openrouter_api_key_env"])
        self._api_key = api_key or os.environ.get(env_key, "")

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(self, doc_path: Path, profile: DocumentProfile) -> ExtractionResult:
        if not _FITZ_OK:
            return ExtractionResult(
                confidence=0.0,
                document=None,
                strategy_name=self.strategy_name,
                escalate=False,
                error="PyMuPDF (fitz) not installed — cannot rasterise PDF pages",
            )
        if not _HTTPX_OK:
            return ExtractionResult(
                confidence=0.0,
                document=None,
                strategy_name=self.strategy_name,
                escalate=False,
                error="httpx not installed — cannot call OpenRouter API",
            )
        if not self._api_key:
            return ExtractionResult(
                confidence=0.0,
                document=None,
                strategy_name=self.strategy_name,
                escalate=False,
                error="OPENROUTER_API_KEY not set — vision extraction unavailable",
            )

        t0 = time.perf_counter()
        text_blocks: list[TextBlock] = []
        tables: list[TableData] = []
        figures: list[FigureBlock] = []
        reading_order: list[str] = []
        warnings: list[str] = []
        total_cost = 0.0
        seq = 0

        try:
            pdf = fitz.open(str(doc_path))
            n_pages = min(len(pdf), self._max_pages)

            for pg_idx in range(n_pages):
                if total_cost >= self._budget_cap:
                    warnings.append(
                        f"Budget cap ${self._budget_cap:.4f} reached at page {pg_idx + 1} — "
                        "remaining pages skipped"
                    )
                    logger.warning("VisionExtractor: budget cap hit at p%d for %s", pg_idx + 1, doc_path.name)
                    break

                page = pdf[pg_idx]
                pg_num = pg_idx + 1
                img_b64 = self._rasterise_page(page)
                if not img_b64:
                    warnings.append(f"p{pg_num}: rasterisation failed — skipped")
                    continue

                pg_data, page_cost, pg_warns = self._call_vlm(img_b64, pg_num)
                total_cost += page_cost
                warnings.extend(pg_warns)

                if pg_data:
                    blks, tbls, figs, order, seq = self._parse_page_response(
                        pg_data, pg_num, profile.doc_id, seq
                    )
                    text_blocks.extend(blks)
                    tables.extend(tbls)
                    figures.extend(figs)
                    reading_order.extend(order)

            pdf.close()
        except Exception as exc:
            logger.error("VisionExtractor failed on %s: %s", doc_path.name, exc)
            return ExtractionResult(
                confidence=0.0,
                document=None,
                strategy_name=self.strategy_name,
                escalate=False,
                error=str(exc),
            )

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
                cost_estimate_usd=round(total_cost, 6),
                processing_time_s=round(elapsed, 3),
                tool_version=self._model,
                warnings=warnings,
            ),
        )
        return ExtractionResult(
            confidence=confidence,
            document=doc,
            strategy_name=self.strategy_name,
            escalate=False,  # Strategy C is the last resort — no further escalation
            warnings=warnings,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rasterise_page(self, page) -> Optional[str]:
        """Render a fitz page to a base-64 encoded PNG string."""
        try:
            mat = fitz.Matrix(self._dpi / 72, self._dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode("utf-8")
        except Exception as exc:
            logger.warning("Rasterisation error: %s", exc)
            return None

    def _call_vlm(
        self, img_b64: str, pg_num: int
    ) -> tuple[Optional[dict], float, list[str]]:
        """
        Call the OpenRouter multimodal API with the page image.
        Returns (parsed_json, cost_usd, warnings).
        """
        warnings: list[str] = []
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        try:
            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/document-intelligence-refinery",
                },
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            in_tokens = usage.get("prompt_tokens", 0)
            out_tokens = usage.get("completion_tokens", 0)
            cost = (in_tokens / 1000) * self._cost_in + (out_tokens / 1000) * self._cost_out

            # Parse JSON from model response
            parsed = json.loads(content)
            return parsed, cost, warnings

        except json.JSONDecodeError as exc:
            warnings.append(f"p{pg_num}: VLM returned non-JSON response — {exc}")
            return None, 0.0, warnings
        except Exception as exc:
            warnings.append(f"p{pg_num}: VLM API error — {exc}")
            return None, 0.0, warnings

    def _parse_page_response(
        self,
        data: dict,
        pg_num: int,
        doc_id: str,
        seq: int,
    ) -> tuple[list[TextBlock], list[TableData], list[FigureBlock], list[str], int]:
        text_blocks: list[TextBlock] = []
        tables: list[TableData] = []
        figures: list[FigureBlock] = []
        reading_order: list[str] = []

        # Paragraphs
        for para in data.get("paragraphs", []):
            blk_id = f"{doc_id}_p{pg_num}_b{seq:05d}"
            bbox_raw = para.get("bbox_estimate", {})
            bbox = BoundingBox(
                x0=bbox_raw.get("x0", 0) * 5.95,  # % of 595pt page width
                y0=bbox_raw.get("y0", 0) * 8.42,
                x1=bbox_raw.get("x1", 100) * 5.95,
                y1=bbox_raw.get("y1", 100) * 8.42,
                page=pg_num,
            )
            text_blocks.append(TextBlock(
                block_id=blk_id,
                text=para.get("text", ""),
                bbox=bbox,
                is_heading=para.get("is_heading", False),
                heading_level=para.get("heading_level"),
                reading_order=seq,
            ))
            reading_order.append(blk_id)
            seq += 1

        # Tables
        for t_idx, tbl in enumerate(data.get("tables", [])):
            tbl_id = f"{doc_id}_p{pg_num}_tbl{t_idx:03d}"
            bbox_raw = tbl.get("bbox_estimate", {})
            bbox = BoundingBox(
                x0=bbox_raw.get("x0", 0) * 5.95,
                y0=bbox_raw.get("y0", 0) * 8.42,
                x1=bbox_raw.get("x1", 100) * 5.95,
                y1=bbox_raw.get("y1", 100) * 8.42,
                page=pg_num,
            )
            headers = tbl.get("headers", [])
            rows = tbl.get("rows", [])
            cells: list[TableCell] = [
                TableCell(row=0, col=c_i, text=h, is_header=True)
                for c_i, h in enumerate(headers)
            ]
            for r_i, row in enumerate(rows, start=1):
                for c_i, val in enumerate(row):
                    cells.append(TableCell(row=r_i, col=c_i, text=str(val)))
            tables.append(TableData(
                table_id=tbl_id,
                bbox=bbox,
                caption=tbl.get("caption"),
                cells=cells,
                col_count=max(len(headers), max((len(r) for r in rows), default=0)),
                row_count=len(rows) + 1,
                extraction_confidence=0.80,
            ))
            reading_order.append(tbl_id)

        # Figures
        for f_idx, fig in enumerate(data.get("figures", [])):
            fig_id = f"{doc_id}_p{pg_num}_fig{f_idx:03d}"
            bbox_raw = fig.get("bbox_estimate", {})
            bbox = BoundingBox(
                x0=bbox_raw.get("x0", 0) * 5.95,
                y0=bbox_raw.get("y0", 0) * 8.42,
                x1=bbox_raw.get("x1", 100) * 5.95,
                y1=bbox_raw.get("y1", 100) * 8.42,
                page=pg_num,
            )
            figures.append(FigureBlock(
                figure_id=fig_id,
                bbox=bbox,
                caption=fig.get("caption"),
                alt_text=fig.get("description"),
            ))
            reading_order.append(fig_id)

        return text_blocks, tables, figures, reading_order, seq

    def _score_confidence(
        self,
        text_blocks: list[TextBlock],
        tables: list[TableData],
        warnings: list[str],
    ) -> float:
        has_content = len(text_blocks) > 0 or len(tables) > 0
        warn_penalty = min(0.30, len([w for w in warnings if "API error" in w]) * 0.10)
        base = 0.85 if has_content else 0.10
        return round(max(0.0, base - warn_penalty), 4)
