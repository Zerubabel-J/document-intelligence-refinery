"""
Stage 1: The Triage Agent — Document Classifier.

Produces a DocumentProfile for every incoming document.  The profile
drives all downstream strategy selection.

Classification pipeline:
  1. origin_type   — character density + image area ratio heuristics
  2. layout_complexity — column heuristics + table/figure bbox analysis
  3. language      — character-set sampling (lightweight, no ML dependency)
  4. domain_hint   — keyword matching against a pluggable keyword map
  5. extraction_cost — decision matrix over the above four dimensions

Thresholds are read from  rubric/extraction_rules.yaml  so they can be
tuned without touching code — a core FDE readiness requirement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed — triage will use stub values")

from src.models.document_profile import (
    ClassificationSignals,
    DocumentProfile,
    DomainHint,
    ExtractionCost,
    LanguageDetection,
    LayoutComplexity,
    OriginType,
)

# ── Default thresholds (overridden by extraction_rules.yaml) ──────────────────
_DEFAULT_THRESHOLDS = {
    "min_chars_per_page_digital": 100,
    "max_image_area_ratio_digital": 0.50,
    "min_char_density_digital": 0.001,
    "table_count_threshold_heavy": 5,
    "figure_count_threshold_heavy": 4,
    "multi_column_bbox_spread": 0.40,
    "pages_to_sample": 5,
}

# ── Domain keyword map (pluggable — can be replaced with VLM classifier) ──────
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "financial": [
        "revenue", "profit", "loss", "balance sheet", "income statement",
        "cash flow", "fiscal", "quarterly", "annual report", "audit",
        "assets", "liabilities", "equity", "dividend", "expenditure",
        "birr", "usd", "budget", "appropriation", "tax",
    ],
    "legal": [
        "whereas", "hereinafter", "pursuant", "indemnify", "arbitration",
        "plaintiff", "defendant", "jurisdiction", "statute", "regulation",
        "act of", "agreement", "contract", "clause", "covenant",
    ],
    "technical": [
        "algorithm", "architecture", "specification", "protocol", "api",
        "infrastructure", "deployment", "microservice", "database", "latency",
        "throughput", "bandwidth", "compliance", "certification",
    ],
    "medical": [
        "patient", "diagnosis", "treatment", "clinical", "dosage",
        "symptom", "pathology", "prognosis", "therapy", "pharmaceutical",
        "adverse", "contraindication", "randomized", "cohort",
    ],
}


class TriageAgent:
    """
    Classifies a PDF document and emits a DocumentProfile.

    Usage::

        agent = TriageAgent(rules_path="rubric/extraction_rules.yaml")
        profile = agent.triage("data/CBE_Annual_Report_2023.pdf")
        profile_path = agent.save_profile(profile)
    """

    def __init__(self, rules_path: str = "rubric/extraction_rules.yaml") -> None:
        self.thresholds = dict(_DEFAULT_THRESHOLDS)
        self._load_rules(rules_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def triage(self, doc_path: str | Path) -> DocumentProfile:
        """Classify the document at *doc_path* and return its DocumentProfile."""
        doc_path = Path(doc_path)
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        doc_id = self._compute_doc_id(doc_path)
        logger.info("Triaging %s  (doc_id=%s…)", doc_path.name, doc_id[:8])

        signals, page_count = self._compute_signals(doc_path)
        origin = self._classify_origin(signals)
        layout = self._classify_layout(signals)
        language = self._detect_language(doc_path)
        domain, keywords = self._classify_domain(doc_path)
        cost = self._estimate_cost(origin, layout)

        return DocumentProfile(
            doc_id=doc_id,
            doc_name=doc_path.name,
            doc_path=str(doc_path.resolve()),
            page_count=page_count,
            origin_type=origin,
            layout_complexity=layout,
            language=language,
            domain_hint=domain,
            estimated_extraction_cost=cost,
            signals=signals,
            domain_keywords_matched=keywords,
        )

    def save_profile(
        self,
        profile: DocumentProfile,
        output_dir: str | Path = ".refinery/profiles",
    ) -> Path:
        """Persist the profile as JSON and return the written path."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{profile.doc_id}.json"
        out_path.write_text(
            profile.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Profile saved → %s", out_path)
        return out_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_rules(self, rules_path: str) -> None:
        path = Path(rules_path)
        if not path.exists():
            logger.warning("Rules file not found at %s — using defaults", rules_path)
            return
        with path.open() as f:
            rules = yaml.safe_load(f)
        triage_cfg = rules.get("triage", {})
        self.thresholds.update(triage_cfg)
        logger.debug("Loaded triage thresholds from %s", rules_path)

    def _compute_doc_id(self, path: Path) -> str:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return sha

    def _compute_signals(self, path: Path) -> tuple[ClassificationSignals, int]:
        """
        Extract heuristic signals from the PDF using pdfplumber.

        Falls back to stub values if pdfplumber is unavailable so the
        rest of the pipeline can still be tested without the dependency.
        """
        if not _PDFPLUMBER_AVAILABLE:
            return self._stub_signals(), 1

        n_sample = int(self.thresholds.get("pages_to_sample", 5))
        char_counts: list[float] = []
        char_densities: list[float] = []
        image_ratios: list[float] = []
        table_count = 0
        figure_count = 0
        font_names: set[str] = set()
        has_text = False
        has_forms = False
        page_count = 0

        try:
            with pdfplumber.open(str(path)) as pdf:
                page_count = len(pdf.pages)
                # Sample evenly across the document
                step = max(1, page_count // n_sample)
                sample_pages = [pdf.pages[i] for i in range(0, page_count, step)][:n_sample]

                for pg in sample_pages:
                    w, h = pg.width or 595, pg.height or 842
                    page_area = w * h

                    # Character stream
                    text = pg.extract_text() or ""
                    chars = pg.chars or []
                    has_text = has_text or bool(chars)
                    char_counts.append(len(chars))
                    density = len(chars) / page_area if page_area > 0 else 0.0
                    char_densities.append(density)

                    # Font inventory
                    for c in chars:
                        fn = c.get("fontname", "")
                        if fn:
                            font_names.add(fn)

                    # Image coverage
                    images = pg.images or []
                    figure_count += len(images)
                    img_area = sum(
                        (im.get("width", 0) * im.get("height", 0)) for im in images
                    )
                    image_ratios.append(img_area / page_area if page_area > 0 else 0.0)

                    # Table heuristic (pdfplumber table detection)
                    try:
                        tables = pg.find_tables()
                        table_count += len(tables)
                    except Exception:
                        pass

                # AcroForm / XFA check
                if pdf.doc.catalog.get("AcroForm"):
                    has_forms = True

        except Exception as exc:
            logger.warning("pdfplumber error on %s: %s — using stub signals", path.name, exc)
            return self._stub_signals(), page_count or 1

        def safe_mean(lst: list) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        signals = ClassificationSignals(
            avg_chars_per_page=safe_mean(char_counts),
            avg_char_density=safe_mean(char_densities),
            image_area_ratio=safe_mean(image_ratios),
            table_count_estimate=table_count,
            figure_count_estimate=figure_count,
            font_count=len(font_names),
            has_embedded_text=has_text,
            has_form_fields=has_forms,
            pages_sampled=len(char_counts),
        )
        return signals, page_count

    @staticmethod
    def _stub_signals() -> ClassificationSignals:
        return ClassificationSignals(
            avg_chars_per_page=0.0,
            avg_char_density=0.0,
            image_area_ratio=0.0,
            table_count_estimate=0,
            figure_count_estimate=0,
            font_count=0,
            has_embedded_text=False,
            has_form_fields=False,
            pages_sampled=0,
        )

    def _classify_origin(self, signals: ClassificationSignals) -> OriginType:
        """
        Decision logic (from DOMAIN_NOTES.md empirical analysis):

        - has_embedded_text=False  → scanned_image  (no character stream at all)
        - avg_chars_per_page < 100 AND image_area_ratio > 0.5 → scanned_image
        - has_form_fields=True → form_fillable
        - avg_chars_per_page > 100 AND image_area_ratio > 0.3 → mixed
        - otherwise → native_digital
        """
        min_chars = float(self.thresholds.get("min_chars_per_page_digital", 100))
        max_img_ratio = float(self.thresholds.get("max_image_area_ratio_digital", 0.50))

        if not signals.has_embedded_text:
            return OriginType.scanned_image
        if signals.avg_chars_per_page < min_chars and signals.image_area_ratio > max_img_ratio:
            return OriginType.scanned_image
        if signals.has_form_fields:
            return OriginType.form_fillable
        if signals.avg_chars_per_page > min_chars and signals.image_area_ratio > 0.30:
            return OriginType.mixed
        return OriginType.native_digital

    def _classify_layout(self, signals: ClassificationSignals) -> LayoutComplexity:
        """
        Heuristic layout classification based on table/figure counts and
        the bbox spread heuristic (multi-column layouts show text blocks
        distributed across a wider horizontal range).

        Thresholds documented in extraction_rules.yaml.
        """
        tbl_threshold = int(self.thresholds.get("table_count_threshold_heavy", 5))
        fig_threshold = int(self.thresholds.get("figure_count_threshold_heavy", 4))

        if signals.table_count_estimate >= tbl_threshold and signals.figure_count_estimate >= fig_threshold:
            return LayoutComplexity.mixed
        if signals.table_count_estimate >= tbl_threshold:
            return LayoutComplexity.table_heavy
        if signals.figure_count_estimate >= fig_threshold:
            return LayoutComplexity.figure_heavy
        if signals.font_count >= 6:
            # Many font families → probable multi-column / complex layout
            return LayoutComplexity.multi_column
        return LayoutComplexity.single_column

    def _detect_language(self, path: Path) -> LanguageDetection:
        """
        Lightweight language detection without ML:
        Sample the first 2 000 characters from extracted text and apply
        character-set heuristics.  Production implementation plugs in
        langdetect or fastText.
        """
        sample_text = ""
        if _PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(str(path)) as pdf:
                    for pg in pdf.pages[:3]:
                        t = pg.extract_text() or ""
                        sample_text += t
                        if len(sample_text) > 2000:
                            break
            except Exception:
                pass

        sample_text = sample_text[:2000]

        # Ethiopic Unicode block: U+1200–U+137F
        ethiopic_chars = sum(1 for c in sample_text if "\u1200" <= c <= "\u137f")
        latin_chars = sum(1 for c in sample_text if c.isalpha() and c.isascii())

        if not sample_text:
            return LanguageDetection(language_code="unknown", confidence=0.0)

        total_alpha = max(1, ethiopic_chars + latin_chars)
        ethiopic_ratio = ethiopic_chars / total_alpha

        if ethiopic_ratio > 0.6:
            return LanguageDetection(language_code="am", confidence=round(ethiopic_ratio, 2))
        if ethiopic_ratio > 0.2:
            return LanguageDetection(
                language_code="en",
                confidence=round(1.0 - ethiopic_ratio, 2),
                secondary_language="am",
            )
        return LanguageDetection(language_code="en", confidence=min(1.0, round(latin_chars / total_alpha, 2)))

    def _classify_domain(self, path: Path) -> tuple[DomainHint, list[str]]:
        """
        Keyword-based domain classification — pluggable strategy pattern.

        Returns the matched domain and the list of keywords that triggered it.
        Falls back to DomainHint.general if no keywords match.
        """
        sample_text = ""
        if _PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(str(path)) as pdf:
                    for pg in pdf.pages[:5]:
                        t = pg.extract_text() or ""
                        sample_text += t.lower()
                        if len(sample_text) > 5000:
                            break
            except Exception:
                pass

        if not sample_text:
            # Fall back to filename-based heuristic
            sample_text = path.stem.lower()

        scores: dict[str, list[str]] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            matched = [kw for kw in keywords if kw in sample_text]
            if matched:
                scores[domain] = matched

        if not scores:
            return DomainHint.general, []

        best_domain = max(scores, key=lambda d: len(scores[d]))
        return DomainHint(best_domain), scores[best_domain]

    def _estimate_cost(self, origin: OriginType, layout: LayoutComplexity) -> ExtractionCost:
        """
        Decision matrix mapping (origin, layout) → extraction cost tier.

        Matrix documented in extraction_rules.yaml under cost_matrix.
        """
        if origin == OriginType.scanned_image:
            return ExtractionCost.needs_vision_model

        if origin == OriginType.native_digital and layout == LayoutComplexity.single_column:
            return ExtractionCost.fast_text_sufficient

        if layout in (LayoutComplexity.table_heavy, LayoutComplexity.multi_column, LayoutComplexity.figure_heavy):
            return ExtractionCost.needs_layout_model

        if layout == LayoutComplexity.mixed or origin == OriginType.mixed:
            return ExtractionCost.needs_layout_model

        return ExtractionCost.fast_text_sufficient
