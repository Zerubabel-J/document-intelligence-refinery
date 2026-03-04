# The Document Intelligence Refinery

A production-grade, multi-stage agentic pipeline that ingests heterogeneous enterprise documents (PDFs, scans, reports, spreadsheets) and emits structured, queryable, spatially-indexed knowledge.

Built for the 10Academy FDE Program — Week 3 Challenge.

---

## Architecture

```
 Input PDF
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Triage Agent                                  │
│  • origin_type detection (digital / scanned / mixed)    │
│  • layout_complexity classification                     │
│  • domain_hint (financial / legal / technical / ...)    │
│  • estimated_extraction_cost tier                       │
│  → Emits: DocumentProfile (.refinery/profiles/)         │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Extraction Router (Confidence-Gated)          │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Strategy A — Fast Text (pdfplumber)  [Low cost] │    │
│  │   confidence < 0.60 → escalate to B             │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ Strategy B — Layout-Aware (Docling)  [Med cost] │    │
│  │   confidence < 0.55 → escalate to C             │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ Strategy C — Vision-Augmented (VLM) [High cost] │    │
│  │   final fallback — no further escalation        │    │
│  └─────────────────────────────────────────────────┘    │
│  → Emits: ExtractedDocument + extraction_ledger.jsonl   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Semantic Chunking Engine [Final submission]   │
│  → Emits: List[LDU] with provenance + content_hash      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: PageIndex Builder [Final submission]          │
│  → Emits: Hierarchical navigation tree (JSON)           │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5: Query Interface Agent [Final submission]      │
│  Tools: pageindex_navigate | semantic_search |          │
│         structured_query                                │
│  → Every answer includes ProvenanceChain (page + bbox)  │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
document-intelligence-refinery/
├── src/
│   ├── models/               # Pydantic schemas (all pipeline contracts)
│   │   ├── document_profile.py   # DocumentProfile — Triage Agent output
│   │   ├── extracted_document.py # ExtractedDocument — Extraction output
│   │   ├── ldu.py                # LDU — Chunking Engine output
│   │   ├── page_index.py         # PageIndex — Navigation tree
│   │   └── provenance.py         # ProvenanceChain — Audit trail
│   ├── agents/
│   │   ├── triage.py         # Stage 1: Triage Agent
│   │   └── extractor.py      # Stage 2: ExtractionRouter
│   └── strategies/
│       ├── base.py           # BaseExtractor interface
│       ├── fast_text.py      # Strategy A: pdfplumber
│       ├── layout_aware.py   # Strategy B: Docling + adapter
│       └── vision_augmented.py # Strategy C: VLM via OpenRouter
├── rubric/
│   └── extraction_rules.yaml # All thresholds, cost matrix, chunking rules
├── tests/
│   ├── test_triage.py        # Triage Agent unit tests
│   └── test_extraction_confidence.py # Confidence scoring unit tests
├── scripts/
│   └── generate_sample_artifacts.py
├── .refinery/
│   ├── profiles/             # DocumentProfile JSON (12 corpus docs)
│   └── extraction_ledger.jsonl
└── pyproject.toml
```

---

## Setup

### Prerequisites

- Python 3.11+
- pyenv (recommended)

### Install

```bash
# Clone the repository
git clone <repo-url>
cd document-intelligence-refinery

# Set Python version
pyenv local 3.11.9

# Install dependencies
pip install -e ".[dev]"
```

For Strategy B (layout-aware), Docling requires additional system dependencies:
```bash
pip install docling
```

For Strategy C (vision-augmented), set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=sk-or-...
```

---

## Usage

### Triage a document

```python
from src.agents.triage import TriageAgent

agent = TriageAgent(rules_path="rubric/extraction_rules.yaml")
profile = agent.triage("data/CBE_Annual_Report_2023-24.pdf")
agent.save_profile(profile)

print(f"Origin: {profile.origin_type}")
print(f"Layout: {profile.layout_complexity}")
print(f"Cost tier: {profile.estimated_extraction_cost}")
```

### Extract a document

```python
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter

# Step 1: Triage
agent = TriageAgent()
profile = agent.triage("data/CBE_Annual_Report_2023-24.pdf")

# Step 2: Route to the right strategy (with escalation guard)
router = ExtractionRouter()
result = router.route("data/CBE_Annual_Report_2023-24.pdf", profile)

print(f"Strategy used: {result.strategy_name}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Tables extracted: {len(result.document.tables)}")
```

### Run unit tests

```bash
python -m pytest tests/ -v
```

### Generate sample artifacts (from corpus analysis)

```bash
python scripts/generate_sample_artifacts.py
```

---

## Configuration

All thresholds, cost matrix entries, and chunking rules are in [rubric/extraction_rules.yaml](rubric/extraction_rules.yaml).

**FDE readiness**: onboarding a new document domain requires modifying only `extraction_rules.yaml` — no Python code changes.

Key configurable parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `triage.min_chars_per_page_digital` | 100 | Below this → likely scanned |
| `triage.max_image_area_ratio_digital` | 0.50 | Above this → likely scanned |
| `strategy_a.confidence_threshold` | 0.60 | Triggers A→B escalation |
| `strategy_b.confidence_threshold` | 0.55 | Triggers B→C escalation |
| `strategy_c.budget_cap_usd` | 0.10 | Max API spend per document |
| `strategy_c.model` | `google/gemini-flash-1.5` | VLM model |

---

## Cost Analysis

| Strategy | Tool | Cost / page | Trigger condition |
|----------|------|-------------|-------------------|
| A — Fast Text | pdfplumber | $0.00 | native_digital + single_column |
| B — Layout-Aware | Docling | $0.00 (local) | multi_column / table_heavy / mixed |
| C — Vision-Augmented | Gemini Flash (OpenRouter) | ~$0.0004 input + $0.0011 output | scanned_image / low confidence |

**Example cost estimates:**
- CBE Annual Report (148 pages, native_digital, Strategy B): **$0.00**
- DBE Audit Report (68 pages, scanned, Strategy C): **~$0.06**
- Mixed document triggering A→B escalation: **$0.00** (Docling is local)

---

## Extraction Ledger

Every extraction is logged to [`.refinery/extraction_ledger.jsonl`](.refinery/extraction_ledger.jsonl):

```jsonl
{
  "timestamp": "2026-03-03T22:00:00+00:00",
  "doc_id": "a1b2c3...",
  "doc_name": "CBE_Annual_Report_2023-24.pdf",
  "strategy_used": "layout_aware",
  "confidence_score": 0.8821,
  "cost_estimate_usd": 0.0,
  "processing_time_s": 14.23,
  "escalation_chain": [],
  "success": true,
  "warnings": [],
  "error": null
}
```

---

## Rubric Coverage (Interim Submission)

| Criterion | Status |
|-----------|--------|
| Core Pydantic Schema Design (15 pts) | `DocumentProfile`, `ExtractedDocument`, `LDU`, `PageIndex`, `ProvenanceChain` — all fully typed |
| Triage Agent — Document Classification (25 pts) | origin_type, layout_complexity, domain_hint, extraction_cost — all implemented with heuristic signals |
| Multi-Strategy Extraction (25 pts) | Strategy A (pdfplumber), B (Docling adapter), C (VLM/OpenRouter) — all with shared interface |
| Extraction Router with Confidence-Gated Escalation (20 pts) | A→B→C escalation guard, ledger logging |
| Externalized Configuration (15 pts) | `rubric/extraction_rules.yaml` — all thresholds, cost matrix, chunking rules |
