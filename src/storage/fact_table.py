"""
FactTable — SQLite-backed structured fact storage.

Extracts key-value facts from financial/numerical documents and stores
them in a queryable SQLite table.  Supports SQL queries like:
  SELECT value FROM facts WHERE key LIKE '%revenue%' AND doc_id = 'abc'

Used by the Query Interface Agent's structured_query tool.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

from src.models.ldu import LDU, ChunkType

logger = logging.getLogger(__name__)

_DB_PATH = Path(".refinery/fact_table.db")

_NUM_PATTERN = re.compile(
    r"[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|%))?",
    re.IGNORECASE,
)

_KV_PATTERNS = [
    # "Revenue: 4.2 billion" or "Revenue  4,200,000"
    re.compile(r"([A-Za-z][\w\s/\-]{2,40}?)\s*[:=]\s*([\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|%)?))"),
    # Table row: "Total Assets | 123,456,789"
    re.compile(r"([A-Za-z][\w\s/\-]{2,40}?)\s*\|\s*([\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|%)?))"),
]


class FactTableDB:
    """SQLite wrapper for the fact table."""

    def __init__(self, db_path: str | Path = _DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                doc_name TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                numeric_value REAL,
                unit TEXT,
                page_number INTEGER,
                chunk_id TEXT,
                section TEXT,
                category_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_facts_doc_id ON facts(doc_id);
            CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
        """)
        self._conn.commit()

    def insert_fact(
        self,
        doc_id: str,
        doc_name: str,
        key: str,
        value: str,
        numeric_value: Optional[float] = None,
        unit: Optional[str] = None,
        page_number: Optional[int] = None,
        chunk_id: Optional[str] = None,
        section: Optional[str] = None,
        category_path: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO facts
               (doc_id, doc_name, key, value, numeric_value, unit, page_number,
                chunk_id, section, category_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, doc_name, key, value, numeric_value, unit,
             page_number, chunk_id, section, category_path),
        )
        self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a SELECT query and return results as list of dicts."""
        try:
            cursor = self._conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            logger.error("SQL query error: %s", exc)
            return []

    def search_facts(self, keyword: str, doc_id: Optional[str] = None) -> list[dict]:
        """Search facts by keyword in key or value columns."""
        if doc_id:
            return self.query(
                "SELECT * FROM facts WHERE (key LIKE ? OR value LIKE ?) AND doc_id = ? ORDER BY page_number",
                (f"%{keyword}%", f"%{keyword}%", doc_id),
            )
        return self.query(
            "SELECT * FROM facts WHERE key LIKE ? OR value LIKE ? ORDER BY doc_name, page_number",
            (f"%{keyword}%", f"%{keyword}%"),
        )

    def close(self) -> None:
        self._conn.close()


class FactTableExtractor:
    """
    Extracts key-value facts from LDUs and inserts them into the FactTableDB.

    Targets table-typed and paragraph-typed chunks from financial documents.
    """

    def __init__(self, db: Optional[FactTableDB] = None) -> None:
        self._db = db or FactTableDB()

    def extract_from_ldus(
        self, ldus: list[LDU], doc_id: str, doc_name: str,
    ) -> int:
        """Extract facts from LDUs and store in DB. Returns count of facts inserted."""
        count = 0
        for ldu in ldus:
            if ldu.chunk_type == ChunkType.table:
                count += self._extract_from_table_chunk(ldu, doc_id, doc_name)
            elif ldu.chunk_type == ChunkType.paragraph:
                count += self._extract_from_text_chunk(ldu, doc_id, doc_name)
        logger.info("Extracted %d facts from %s", count, doc_name)
        return count

    def _extract_from_table_chunk(self, ldu: LDU, doc_id: str, doc_name: str) -> int:
        """Parse markdown table content to extract key-value pairs."""
        lines = ldu.content.strip().split("\n")
        count = 0

        # Find header and data rows
        headers: list[str] = []
        data_rows: list[list[str]] = []
        for line in lines:
            if line.startswith("Caption:"):
                continue
            if "|" in line:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if all(c in ("---", "---", "") for c in cells):
                    continue  # separator row
                if not headers:
                    headers = cells
                else:
                    data_rows.append(cells)

        # Extract facts from each data row
        for row in data_rows:
            if not row:
                continue
            row_label = row[0] if row else ""
            for i, val in enumerate(row[1:], start=1):
                val = val.strip()
                if not val or val == "-":
                    continue
                numeric = self._parse_numeric(val)
                col_header = headers[i] if i < len(headers) else f"col_{i}"
                key = f"{row_label} — {col_header}".strip(" —")
                self._db.insert_fact(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    key=key,
                    value=val,
                    numeric_value=numeric,
                    unit=self._detect_unit(val),
                    page_number=ldu.page_refs[0] if ldu.page_refs else None,
                    chunk_id=ldu.chunk_id,
                    section=ldu.parent_section,
                    category_path=row_label,
                )
                count += 1
        return count

    def _extract_from_text_chunk(self, ldu: LDU, doc_id: str, doc_name: str) -> int:
        """Extract key-value pairs from narrative text."""
        count = 0
        for pattern in _KV_PATTERNS:
            for match in pattern.finditer(ldu.content):
                key = match.group(1).strip()
                value = match.group(2).strip()
                if len(key) < 3 or not value:
                    continue
                numeric = self._parse_numeric(value)
                self._db.insert_fact(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    key=key,
                    value=value,
                    numeric_value=numeric,
                    unit=self._detect_unit(value),
                    page_number=ldu.page_refs[0] if ldu.page_refs else None,
                    chunk_id=ldu.chunk_id,
                    section=ldu.parent_section,
                )
                count += 1
        return count

    @staticmethod
    def _parse_numeric(val: str) -> Optional[float]:
        """Parse a numeric value from a string, handling commas and suffixes."""
        clean = re.sub(r'[,\s]', '', val)
        clean = re.sub(r'(?:billion|million|thousand|%)', '', clean, flags=re.IGNORECASE).strip()
        try:
            num = float(clean)
            lower = val.lower()
            if "billion" in lower:
                num *= 1e9
            elif "million" in lower:
                num *= 1e6
            elif "thousand" in lower:
                num *= 1e3
            return num
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _detect_unit(val: str) -> Optional[str]:
        lower = val.lower()
        if "%" in val:
            return "%"
        if "billion" in lower:
            return "billion"
        if "million" in lower:
            return "million"
        if "birr" in lower or "etb" in lower:
            return "ETB"
        if "$" in val or "usd" in lower:
            return "USD"
        return None
