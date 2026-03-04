"""
Stage 2: The ExtractionRouter — confidence-gated strategy dispatcher.

Reads the DocumentProfile produced by the Triage Agent and delegates to
the appropriate extraction strategy.  Implements the Escalation Guard:

  Strategy A → confidence check → if LOW → escalate to Strategy B
  Strategy B → confidence check → if LOW → escalate to Strategy C
  Strategy C → final — no further escalation

Every extraction is logged to  .refinery/extraction_ledger.jsonl  with:
  doc_id, doc_name, strategy_used, confidence_score, cost_estimate_usd,
  processing_time_s, escalation_chain, warnings, timestamp.

Thresholds are loaded from  rubric/extraction_rules.yaml.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from src.models.document_profile import DocumentProfile, ExtractionCost
from src.models.extracted_document import ExtractedDocument
from src.strategies.base import ExtractionResult
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_aware import LayoutExtractor
from src.strategies.vision_augmented import VisionExtractor

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = {
    "strategy_a": {"confidence_threshold": 0.60},
    "strategy_b": {"confidence_threshold": 0.55},
    "strategy_c": {"confidence_threshold": 0.50},
    "budget_cap_usd": 0.10,
}

_LEDGER_PATH = Path(".refinery/extraction_ledger.jsonl")


class ExtractionRouter:
    """
    Routes each document to the correct extraction strategy based on its
    DocumentProfile and enforces confidence-gated escalation.

    Usage::

        router = ExtractionRouter(rules_path="rubric/extraction_rules.yaml")
        result = router.route(doc_path, profile)
        # result.document is the normalised ExtractedDocument
    """

    def __init__(
        self,
        rules_path: str = "rubric/extraction_rules.yaml",
        ledger_path: str | Path = _LEDGER_PATH,
        api_key: Optional[str] = None,
    ) -> None:
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        self._load_rules(rules_path)
        self._ledger_path = Path(ledger_path)
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

        # Instantiate strategies with loaded thresholds
        self._strategy_a = FastTextExtractor(
            thresholds=self._thresholds.get("strategy_a", {})
        )
        self._strategy_b = LayoutExtractor(
            thresholds=self._thresholds.get("strategy_b", {})
        )
        self._strategy_c = VisionExtractor(
            thresholds=self._thresholds.get("strategy_c", {}),
            api_key=self._api_key,
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def route(
        self,
        doc_path: str | Path,
        profile: DocumentProfile,
    ) -> ExtractionResult:
        """
        Select and execute the extraction strategy for *doc_path*.

        Returns the final ExtractionResult (after any escalation steps).
        Also writes a ledger entry to extraction_ledger.jsonl.
        """
        doc_path = Path(doc_path)
        escalation_chain: list[str] = []
        final_result: Optional[ExtractionResult] = None

        # ── Determine starting strategy ────────────────────────────────────────
        start_cost = profile.estimated_extraction_cost

        if start_cost == ExtractionCost.fast_text_sufficient:
            # Try A → B → C
            result_a = self._run(self._strategy_a, doc_path, profile, escalation_chain)
            if not result_a.escalate and result_a.success:
                final_result = result_a
            else:
                logger.info("Escalating %s from Strategy A to B", doc_path.name)
                escalation_chain.append("A→B")
                result_b = self._run(self._strategy_b, doc_path, profile, escalation_chain)
                if not result_b.escalate and result_b.success:
                    final_result = result_b
                else:
                    logger.info("Escalating %s from Strategy B to C", doc_path.name)
                    escalation_chain.append("B→C")
                    final_result = self._run(self._strategy_c, doc_path, profile, escalation_chain)

        elif start_cost == ExtractionCost.needs_layout_model:
            # Start at B, escalate to C if needed
            result_b = self._run(self._strategy_b, doc_path, profile, escalation_chain)
            if not result_b.escalate and result_b.success:
                final_result = result_b
            else:
                logger.info("Escalating %s from Strategy B to C", doc_path.name)
                escalation_chain.append("B→C")
                final_result = self._run(self._strategy_c, doc_path, profile, escalation_chain)

        else:  # needs_vision_model — scanned document, go straight to C
            escalation_chain.append("direct→C")
            final_result = self._run(self._strategy_c, doc_path, profile, escalation_chain)

        if final_result is None:
            final_result = ExtractionResult(
                confidence=0.0,
                document=None,
                strategy_name="none",
                error="All extraction strategies failed",
            )

        self._write_ledger(profile, final_result, escalation_chain)
        return final_result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(
        self,
        strategy,
        doc_path: Path,
        profile: DocumentProfile,
        escalation_chain: list[str],
    ) -> ExtractionResult:
        logger.info(
            "Running %s on %s", strategy.strategy_name, doc_path.name
        )
        try:
            return strategy.extract(doc_path, profile)
        except Exception as exc:
            logger.error("Strategy %s crashed: %s", strategy.strategy_name, exc)
            return ExtractionResult(
                confidence=0.0,
                document=None,
                strategy_name=strategy.strategy_name,
                escalate=True,
                error=str(exc),
            )

    def _write_ledger(
        self,
        profile: DocumentProfile,
        result: ExtractionResult,
        escalation_chain: list[str],
    ) -> None:
        """Append one JSONL entry to the extraction ledger."""
        cost_usd = 0.0
        proc_time = 0.0
        if result.document:
            cost_usd = result.document.metadata.cost_estimate_usd
            proc_time = result.document.metadata.processing_time_s

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": profile.doc_id,
            "doc_name": profile.doc_name,
            "strategy_used": result.strategy_name,
            "confidence_score": round(result.confidence, 4),
            "cost_estimate_usd": round(cost_usd, 6),
            "processing_time_s": round(proc_time, 3),
            "escalation_chain": escalation_chain,
            "success": result.success,
            "warnings": result.warnings[:10],  # cap ledger verbosity
            "error": result.error,
        }
        try:
            with self._ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.error("Failed to write ledger entry: %s", exc)

    def _load_rules(self, rules_path: str) -> None:
        path = Path(rules_path)
        if not path.exists():
            return
        with path.open() as f:
            rules = yaml.safe_load(f)
        for key in ("strategy_a", "strategy_b", "strategy_c"):
            if key in rules:
                self._thresholds[key] = rules[key]
        if "budget_cap_usd" in rules:
            self._thresholds["budget_cap_usd"] = rules["budget_cap_usd"]
