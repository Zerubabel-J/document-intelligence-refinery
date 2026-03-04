"""
Base extractor interface — the contract all three strategies must satisfy.

Design: Strategy Pattern.
  - Every strategy implements extract() and returns an ExtractionResult.
  - The ExtractionRouter calls extract() without knowing which concrete
    strategy it is using; it only inspects ExtractionResult.confidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument


@dataclass
class ExtractionResult:
    """
    Wrapper returned by every extraction strategy.

    confidence  — composite score in [0, 1].  The ExtractionRouter uses
                  this to decide whether to escalate to a higher-cost strategy.
    document    — the normalised ExtractedDocument (None on hard failure).
    escalate    — True if the strategy itself determined escalation is needed.
    """
    confidence: float
    document: Optional[ExtractedDocument]
    strategy_name: str
    escalate: bool = False
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.document is not None and self.error is None


class BaseExtractor(ABC):
    """Abstract base class for all extraction strategies."""

    #: Strategy identifier used in ledger entries and log messages.
    strategy_name: str = "base"

    #: Confidence threshold below which the strategy should request escalation.
    confidence_threshold: float = 0.6

    def __init__(self, confidence_threshold: Optional[float] = None) -> None:
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold

    @abstractmethod
    def extract(
        self,
        doc_path: Path,
        profile: DocumentProfile,
    ) -> ExtractionResult:
        """
        Extract content from *doc_path* given the pre-computed *profile*.

        Must return an ExtractionResult regardless of success or failure.
        Never raise exceptions — catch internally and set result.error.
        """

    def _should_escalate(self, confidence: float) -> bool:
        return confidence < self.confidence_threshold
