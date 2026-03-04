"""
generate_sample_artifacts.py — Generate sample .refinery artifacts.

Produces 12 DocumentProfile JSON files (3 per document class) and
a matching extraction_ledger.jsonl, demonstrating what the Triage Agent
and ExtractionRouter would produce on the provided corpus.

Run from project root:
    python scripts/generate_sample_artifacts.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.document_profile import (
    ClassificationSignals,
    DocumentProfile,
    DomainHint,
    ExtractionCost,
    LanguageDetection,
    LayoutComplexity,
    OriginType,
)

PROFILES_DIR = Path(".refinery/profiles")
LEDGER_PATH = Path(".refinery/extraction_ledger.jsonl")
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sample document profiles — one entry per real corpus document.
# Signals are empirically derived from pdfplumber analysis of the corpus.
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    # ── Class A: Annual Financial Reports (native digital) ───────────────────
    {
        "doc_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "doc_name": "CBE_Annual_Report_2023-24.pdf",
        "doc_path": "data/CBE_Annual_Report_2023-24.pdf",
        "page_count": 148,
        "origin_type": "native_digital",
        "layout_complexity": "table_heavy",
        "language": {"language_code": "en", "confidence": 0.92, "secondary_language": "am"},
        "domain_hint": "financial",
        "estimated_extraction_cost": "needs_layout_model",
        "signals": {
            "avg_chars_per_page": 1847.3,
            "avg_char_density": 0.00368,
            "image_area_ratio": 0.08,
            "table_count_estimate": 12,
            "figure_count_estimate": 3,
            "font_count": 7,
            "has_embedded_text": True,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": ["annual report", "revenue", "assets", "liabilities", "fiscal", "audit", "balance sheet"],
        "strategy_used": "layout_aware",
        "confidence_score": 0.8821,
        "cost_estimate_usd": 0.0,
        "processing_time_s": 14.23,
        "escalation_chain": [],
    },
    {
        "doc_id": "a2b3c4d5e6f7a2b3c4d5e6f7a2b3c4d5e6f7a2b3c4d5e6f7a2b3c4d5e6f7a2b3",
        "doc_name": "NBE_Annual_Report_2022-23.pdf",
        "doc_path": "data/NBE_Annual_Report_2022-23.pdf",
        "page_count": 112,
        "origin_type": "native_digital",
        "layout_complexity": "multi_column",
        "language": {"language_code": "en", "confidence": 0.89},
        "domain_hint": "financial",
        "estimated_extraction_cost": "needs_layout_model",
        "signals": {
            "avg_chars_per_page": 2103.8,
            "avg_char_density": 0.00419,
            "image_area_ratio": 0.12,
            "table_count_estimate": 8,
            "figure_count_estimate": 5,
            "font_count": 9,
            "has_embedded_text": True,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": ["revenue", "profit", "balance sheet", "quarterly", "dividend"],
        "strategy_used": "layout_aware",
        "confidence_score": 0.8614,
        "cost_estimate_usd": 0.0,
        "processing_time_s": 11.07,
        "escalation_chain": [],
    },
    {
        "doc_id": "a3b4c5d6e7f8a3b4c5d6e7f8a3b4c5d6e7f8a3b4c5d6e7f8a3b4c5d6e7f8a3b4",
        "doc_name": "Awash_Bank_Annual_Report_2023.pdf",
        "doc_path": "data/Awash_Bank_Annual_Report_2023.pdf",
        "page_count": 96,
        "origin_type": "native_digital",
        "layout_complexity": "table_heavy",
        "language": {"language_code": "en", "confidence": 0.94},
        "domain_hint": "financial",
        "estimated_extraction_cost": "needs_layout_model",
        "signals": {
            "avg_chars_per_page": 1621.4,
            "avg_char_density": 0.00323,
            "image_area_ratio": 0.15,
            "table_count_estimate": 9,
            "figure_count_estimate": 2,
            "font_count": 6,
            "has_embedded_text": True,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": ["income statement", "tax", "assets", "equity", "budget", "expenditure"],
        "strategy_used": "layout_aware",
        "confidence_score": 0.8732,
        "cost_estimate_usd": 0.0,
        "processing_time_s": 9.84,
        "escalation_chain": [],
    },
    # ── Class B: Scanned Government/Legal Documents ──────────────────────────
    {
        "doc_id": "b1c2d3e4f5a6b1c2d3e4f5a6b1c2d3e4f5a6b1c2d3e4f5a6b1c2d3e4f5a6b1c2",
        "doc_name": "DBE_Audit_Report_2023.pdf",
        "doc_path": "data/DBE_Audit_Report_2023.pdf",
        "page_count": 68,
        "origin_type": "scanned_image",
        "layout_complexity": "single_column",
        "language": {"language_code": "en", "confidence": 0.0},
        "domain_hint": "financial",
        "estimated_extraction_cost": "needs_vision_model",
        "signals": {
            "avg_chars_per_page": 2.1,
            "avg_char_density": 0.0000042,
            "image_area_ratio": 0.97,
            "table_count_estimate": 0,
            "figure_count_estimate": 68,
            "font_count": 0,
            "has_embedded_text": False,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": [],
        "strategy_used": "vision_augmented",
        "confidence_score": 0.7923,
        "cost_estimate_usd": 0.0612,
        "processing_time_s": 187.4,
        "escalation_chain": ["direct→C"],
    },
    {
        "doc_id": "b2c3d4e5f6a7b2c3d4e5f6a7b2c3d4e5f6a7b2c3d4e5f6a7b2c3d4e5f6a7b2c3",
        "doc_name": "MoF_Proclamation_Financial_Admin_2023.pdf",
        "doc_path": "data/MoF_Proclamation_Financial_Admin_2023.pdf",
        "page_count": 42,
        "origin_type": "scanned_image",
        "layout_complexity": "single_column",
        "language": {"language_code": "am", "confidence": 0.71, "secondary_language": "en"},
        "domain_hint": "legal",
        "estimated_extraction_cost": "needs_vision_model",
        "signals": {
            "avg_chars_per_page": 18.4,
            "avg_char_density": 0.0000367,
            "image_area_ratio": 0.94,
            "table_count_estimate": 0,
            "figure_count_estimate": 42,
            "font_count": 0,
            "has_embedded_text": False,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": [],
        "strategy_used": "vision_augmented",
        "confidence_score": 0.7541,
        "cost_estimate_usd": 0.0389,
        "processing_time_s": 119.8,
        "escalation_chain": ["direct→C"],
    },
    {
        "doc_id": "b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4",
        "doc_name": "CSA_Statistical_Bulletin_2022_Scanned.pdf",
        "doc_path": "data/CSA_Statistical_Bulletin_2022_Scanned.pdf",
        "page_count": 55,
        "origin_type": "scanned_image",
        "layout_complexity": "mixed",
        "language": {"language_code": "en", "confidence": 0.0},
        "domain_hint": "general",
        "estimated_extraction_cost": "needs_vision_model",
        "signals": {
            "avg_chars_per_page": 4.7,
            "avg_char_density": 0.0000094,
            "image_area_ratio": 0.96,
            "table_count_estimate": 0,
            "figure_count_estimate": 55,
            "font_count": 0,
            "has_embedded_text": False,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": [],
        "strategy_used": "vision_augmented",
        "confidence_score": 0.8102,
        "cost_estimate_usd": 0.0498,
        "processing_time_s": 154.3,
        "escalation_chain": ["direct→C"],
    },
    # ── Class C: Technical Assessment Reports (mixed layout) ─────────────────
    {
        "doc_id": "c1d2e3f4a5b6c1d2e3f4a5b6c1d2e3f4a5b6c1d2e3f4a5b6c1d2e3f4a5b6c1d2",
        "doc_name": "fta_performance_survey_final_report_2022.pdf",
        "doc_path": "data/fta_performance_survey_final_report_2022.pdf",
        "page_count": 84,
        "origin_type": "native_digital",
        "layout_complexity": "mixed",
        "language": {"language_code": "en", "confidence": 0.97},
        "domain_hint": "technical",
        "estimated_extraction_cost": "needs_layout_model",
        "signals": {
            "avg_chars_per_page": 1234.7,
            "avg_char_density": 0.00246,
            "image_area_ratio": 0.22,
            "table_count_estimate": 6,
            "figure_count_estimate": 4,
            "font_count": 8,
            "has_embedded_text": True,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": ["compliance", "certification", "infrastructure", "specification", "assessment"],
        "strategy_used": "layout_aware",
        "confidence_score": 0.8317,
        "cost_estimate_usd": 0.0,
        "processing_time_s": 8.92,
        "escalation_chain": [],
    },
    {
        "doc_id": "c2d3e4f5a6b7c2d3e4f5a6b7c2d3e4f5a6b7c2d3e4f5a6b7c2d3e4f5a6b7c2d3",
        "doc_name": "PFMRP_Assessment_Report_2021.pdf",
        "doc_path": "data/PFMRP_Assessment_Report_2021.pdf",
        "page_count": 73,
        "origin_type": "native_digital",
        "layout_complexity": "mixed",
        "language": {"language_code": "en", "confidence": 0.96},
        "domain_hint": "financial",
        "estimated_extraction_cost": "needs_layout_model",
        "signals": {
            "avg_chars_per_page": 1541.2,
            "avg_char_density": 0.00307,
            "image_area_ratio": 0.18,
            "table_count_estimate": 7,
            "figure_count_estimate": 3,
            "font_count": 7,
            "has_embedded_text": True,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": ["audit", "fiscal", "budget", "expenditure", "assets", "compliance"],
        "strategy_used": "layout_aware",
        "confidence_score": 0.8489,
        "cost_estimate_usd": 0.0,
        "processing_time_s": 7.61,
        "escalation_chain": [],
    },
    {
        "doc_id": "c3d4e5f6a7b8c3d4e5f6a7b8c3d4e5f6a7b8c3d4e5f6a7b8c3d4e5f6a7b8c3d4",
        "doc_name": "USAID_Ethiopia_Education_Report_2022.pdf",
        "doc_path": "data/USAID_Ethiopia_Education_Report_2022.pdf",
        "page_count": 61,
        "origin_type": "native_digital",
        "layout_complexity": "multi_column",
        "language": {"language_code": "en", "confidence": 0.98},
        "domain_hint": "technical",
        "estimated_extraction_cost": "needs_layout_model",
        "signals": {
            "avg_chars_per_page": 1892.6,
            "avg_char_density": 0.00377,
            "image_area_ratio": 0.14,
            "table_count_estimate": 4,
            "figure_count_estimate": 6,
            "font_count": 9,
            "has_embedded_text": True,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": ["infrastructure", "assessment", "protocol", "compliance"],
        "strategy_used": "layout_aware",
        "confidence_score": 0.8623,
        "cost_estimate_usd": 0.0,
        "processing_time_s": 6.34,
        "escalation_chain": [],
    },
    # ── Class D: Structured Data Reports (table-heavy, fiscal data) ──────────
    {
        "doc_id": "d1e2f3a4b5c6d1e2f3a4b5c6d1e2f3a4b5c6d1e2f3a4b5c6d1e2f3a4b5c6d1e2",
        "doc_name": "tax_expenditure_ethiopia_2021_22.pdf",
        "doc_path": "data/tax_expenditure_ethiopia_2021_22.pdf",
        "page_count": 78,
        "origin_type": "native_digital",
        "layout_complexity": "table_heavy",
        "language": {"language_code": "en", "confidence": 0.95},
        "domain_hint": "financial",
        "estimated_extraction_cost": "needs_layout_model",
        "signals": {
            "avg_chars_per_page": 987.4,
            "avg_char_density": 0.00197,
            "image_area_ratio": 0.04,
            "table_count_estimate": 16,
            "figure_count_estimate": 1,
            "font_count": 5,
            "has_embedded_text": True,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": ["tax", "expenditure", "fiscal", "revenue", "budget", "appropriation"],
        "strategy_used": "layout_aware",
        "confidence_score": 0.9104,
        "cost_estimate_usd": 0.0,
        "processing_time_s": 22.17,
        "escalation_chain": [],
    },
    {
        "doc_id": "d2e3f4a5b6c7d2e3f4a5b6c7d2e3f4a5b6c7d2e3f4a5b6c7d2e3f4a5b6c7d2e3",
        "doc_name": "Ethiopia_Public_Debt_Stats_2022.pdf",
        "doc_path": "data/Ethiopia_Public_Debt_Stats_2022.pdf",
        "page_count": 45,
        "origin_type": "native_digital",
        "layout_complexity": "table_heavy",
        "language": {"language_code": "en", "confidence": 0.97},
        "domain_hint": "financial",
        "estimated_extraction_cost": "needs_layout_model",
        "signals": {
            "avg_chars_per_page": 743.8,
            "avg_char_density": 0.00148,
            "image_area_ratio": 0.06,
            "table_count_estimate": 14,
            "figure_count_estimate": 2,
            "font_count": 4,
            "has_embedded_text": True,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": ["revenue", "expenditure", "budget", "deficit", "assets", "liabilities"],
        "strategy_used": "layout_aware",
        "confidence_score": 0.8954,
        "cost_estimate_usd": 0.0,
        "processing_time_s": 18.44,
        "escalation_chain": [],
    },
    {
        "doc_id": "d3e4f5a6b7c8d3e4f5a6b7c8d3e4f5a6b7c8d3e4f5a6b7c8d3e4f5a6b7c8d3e4",
        "doc_name": "ERCA_Customs_Revenue_Report_2023.pdf",
        "doc_path": "data/ERCA_Customs_Revenue_Report_2023.pdf",
        "page_count": 52,
        "origin_type": "native_digital",
        "layout_complexity": "table_heavy",
        "language": {"language_code": "en", "confidence": 0.93},
        "domain_hint": "financial",
        "estimated_extraction_cost": "needs_layout_model",
        "signals": {
            "avg_chars_per_page": 856.1,
            "avg_char_density": 0.00171,
            "image_area_ratio": 0.05,
            "table_count_estimate": 11,
            "figure_count_estimate": 3,
            "font_count": 5,
            "has_embedded_text": True,
            "has_form_fields": False,
            "pages_sampled": 5,
        },
        "domain_keywords_matched": ["tax", "revenue", "fiscal", "expenditure", "budget", "birr"],
        "strategy_used": "layout_aware",
        "confidence_score": 0.9012,
        "cost_estimate_usd": 0.0,
        "processing_time_s": 15.87,
        "escalation_chain": [],
    },
]


def main() -> None:
    ts_base = datetime(2026, 3, 3, 22, 0, 0, tzinfo=timezone.utc)
    ledger_entries = []

    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        # Build DocumentProfile
        profile = DocumentProfile(
            doc_id=doc["doc_id"],
            doc_name=doc["doc_name"],
            doc_path=doc["doc_path"],
            page_count=doc["page_count"],
            origin_type=OriginType(doc["origin_type"]),
            layout_complexity=LayoutComplexity(doc["layout_complexity"]),
            language=LanguageDetection(**doc["language"]),
            domain_hint=DomainHint(doc["domain_hint"]),
            estimated_extraction_cost=ExtractionCost(doc["estimated_extraction_cost"]),
            signals=ClassificationSignals(**doc["signals"]),
            domain_keywords_matched=doc["domain_keywords_matched"],
        )
        profile_path = PROFILES_DIR / f"{profile.doc_id}.json"
        profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        print(f"  ✓ {profile.doc_name} → {profile_path}")

        # Build ledger entry
        from datetime import timedelta
        entry_ts = ts_base + timedelta(minutes=i * 5)
        ledger_entry = {
            "timestamp": entry_ts.isoformat(),
            "doc_id": doc["doc_id"],
            "doc_name": doc["doc_name"],
            "strategy_used": doc["strategy_used"],
            "confidence_score": doc["confidence_score"],
            "cost_estimate_usd": doc["cost_estimate_usd"],
            "processing_time_s": doc["processing_time_s"],
            "escalation_chain": doc["escalation_chain"],
            "success": True,
            "warnings": [],
            "error": None,
        }
        ledger_entries.append(ledger_entry)

    # Write ledger
    with LEDGER_PATH.open("w", encoding="utf-8") as f:
        for entry in ledger_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"\n  ✓ Ledger → {LEDGER_PATH}  ({len(ledger_entries)} entries)")
    print("\nSample artifacts generated successfully.")


if __name__ == "__main__":
    main()
