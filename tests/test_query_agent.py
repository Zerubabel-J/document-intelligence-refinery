"""
Unit tests for Query Interface Agent and Audit System.

Tests cover:
1. QueryTools — pageindex_navigate, semantic_search, structured_query
2. AuditEngine — claim verification, provenance chain building
3. FactTableDB — SQLite fact storage and retrieval
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from src.agents.auditor import AuditEngine
from src.agents.query_agent import QueryTools
from src.models.page_index import PageIndex, PageIndexNode
from src.storage.fact_table import FactTableDB, FactTableExtractor
from src.models.ldu import LDU, ChunkType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_page_index():
    root = PageIndexNode(
        node_id="doc_sec_1_0000",
        title="Revenue Analysis",
        page_start=1,
        page_end=10,
        depth=1,
        summary="This section covers revenue data for FY 2023.",
        key_entities=["Revenue", "FY 2023"],
        data_types_present=["table"],
        chunk_ids=["doc_000001", "doc_000002"],
    )
    child = PageIndexNode(
        node_id="doc_sec_2_0001",
        title="Quarterly Breakdown",
        page_start=5,
        page_end=10,
        depth=2,
        summary="Quarterly revenue figures.",
        key_entities=["Q1", "Q2", "Q3", "Q4"],
        chunk_ids=["doc_000003"],
        parent_node_id="doc_sec_1_0000",
    )
    root.child_nodes = [child]
    return PageIndex(
        doc_id="doc",
        doc_name="test.pdf",
        root_nodes=[root],
        total_sections=2,
    )


@pytest.fixture()
def fact_db(tmp_path):
    db = FactTableDB(db_path=tmp_path / "test_facts.db")
    db.insert_fact(
        doc_id="doc", doc_name="test.pdf",
        key="Total Revenue", value="4,200,000,000",
        numeric_value=4.2e9, unit="ETB",
        page_number=5, chunk_id="doc_000002",
        section="Revenue Analysis",
    )
    db.insert_fact(
        doc_id="doc", doc_name="test.pdf",
        key="Operating Expenses", value="1,800,000,000",
        numeric_value=1.8e9, unit="ETB",
        page_number=8, chunk_id="doc_000004",
        section="Expenditure",
    )
    return db


# ── QueryTools tests ──────────────────────────────────────────────────────────

class TestQueryTools:

    def test_pageindex_navigate(self, sample_page_index):
        tools = QueryTools(page_indexes={"doc": sample_page_index})
        results = tools.pageindex_navigate("Revenue", doc_id="doc")
        assert len(results) >= 1
        assert results[0]["section_title"] == "Revenue Analysis"

    def test_pageindex_navigate_nested(self, sample_page_index):
        tools = QueryTools(page_indexes={"doc": sample_page_index})
        results = tools.pageindex_navigate("Quarterly", doc_id="doc")
        assert any(r["section_title"] == "Quarterly Breakdown" for r in results)

    def test_structured_query_keyword(self, fact_db):
        tools = QueryTools(page_indexes={}, fact_db=fact_db)
        results = tools.structured_query("revenue", doc_id="doc")
        assert len(results) >= 1
        assert any("Revenue" in r.get("key", "") for r in results)

    def test_structured_query_sql(self, fact_db):
        tools = QueryTools(page_indexes={}, fact_db=fact_db)
        results = tools.structured_query(
            "SQL: SELECT key, value FROM facts WHERE doc_id = 'doc'"
        )
        assert len(results) == 2


# ── FactTable tests ───────────────────────────────────────────────────────────

class TestFactTable:

    def test_insert_and_search(self, fact_db):
        results = fact_db.search_facts("Revenue")
        assert len(results) >= 1
        assert results[0]["numeric_value"] == 4.2e9

    def test_search_by_doc_id(self, fact_db):
        results = fact_db.search_facts("Revenue", doc_id="doc")
        assert len(results) >= 1
        results_none = fact_db.search_facts("Revenue", doc_id="nonexistent")
        assert len(results_none) == 0

    def test_sql_query(self, fact_db):
        results = fact_db.query(
            "SELECT SUM(numeric_value) as total FROM facts WHERE doc_id = ?",
            ("doc",)
        )
        assert results[0]["total"] == 6.0e9  # 4.2B + 1.8B


class TestFactTableExtractor:

    def test_extract_from_table_chunk(self, tmp_path):
        db = FactTableDB(db_path=tmp_path / "ext_facts.db")
        extractor = FactTableExtractor(db=db)

        ldu = LDU(
            chunk_id="doc_000010", doc_id="doc", sequence=10,
            content="| Metric | FY 2023 |\n|---|---|\n| Revenue | 4,200 |\n| Expenses | 1,800 |",
            chunk_type=ChunkType.table,
            token_count=20, page_refs=[5],
            parent_section="Financial Summary",
            content_hash=hashlib.sha256(b"test").hexdigest(),
        )
        count = extractor.extract_from_ldus([ldu], doc_id="doc", doc_name="test.pdf")
        assert count >= 2

        results = db.search_facts("Revenue")
        assert len(results) >= 1
        db.close()


# ── Audit Engine tests ────────────────────────────────────────────────────────

class TestAuditEngine:

    def test_verify_claim_with_facts(self, fact_db):
        engine = AuditEngine(fact_db=fact_db)
        chain = engine.verify_claim("Total Revenue was 4,200,000,000", doc_id="doc")
        assert len(chain.sources) >= 1
        assert chain.confidence > 0

    def test_unverifiable_claim(self, tmp_path):
        db = FactTableDB(db_path=tmp_path / "empty.db")
        engine = AuditEngine(fact_db=db)
        chain = engine.verify_claim("The GDP was 500 trillion")
        assert chain.unverifiable_flag is True
        assert chain.is_verified is False
        db.close()

    def test_provenance_chain_structure(self, fact_db):
        engine = AuditEngine(fact_db=fact_db)
        chain = engine.build_provenance_for_answer(
            query="What is total revenue?",
            answer="4.2 billion birr",
            supporting_chunks=[{
                "chunk_id": "doc_000002",
                "content": "Total Revenue: 4,200,000,000 ETB",
                "metadata": {
                    "doc_id": "doc",
                    "page_refs": "5",
                    "parent_section": "Revenue Analysis",
                    "content_hash": "abc123",
                },
            }],
        )
        assert chain.query == "What is total revenue?"
        assert len(chain.sources) == 1
        assert chain.sources[0].page_number == 5
        assert chain.is_verified is True
