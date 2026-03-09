"""
Stage 4: PageIndex Builder.

Builds a hierarchical navigation tree over a document — the equivalent
of a "smart table of contents" that an LLM can traverse to locate
information without reading the entire document.

The PageIndex tree is built from heading-typed LDUs.  Each section node
gets an LLM-generated summary (2-3 sentences) using a fast, cheap model.

Output is stored at  .refinery/pageindex/{doc_id}.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.ldu import LDU, ChunkType
from src.models.page_index import PageIndex, PageIndexNode

logger = logging.getLogger(__name__)

# ── Optional LLM import for summaries ─────────────────────────────────────────
try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False

_ENTITY_KEYWORDS = [
    "bank", "ministry", "authority", "commission", "agency",
    "ethiopia", "birr", "usd", "revenue", "expenditure",
    "q1", "q2", "q3", "q4", "fy", "fiscal",
]


class PageIndexBuilder:
    """
    Builds a PageIndex tree from a list of LDUs.

    Usage::

        builder = PageIndexBuilder()
        index = builder.build(ldus, doc_id="abc", doc_name="report.pdf")
        builder.save(index)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-flash-1.5",
        enable_llm_summaries: bool = True,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._model = model
        self._enable_summaries = enable_llm_summaries and bool(self._api_key) and _HTTPX_OK

    def build(self, ldus: list[LDU], doc_id: str, doc_name: str) -> PageIndex:
        """Build a PageIndex tree from a list of LDUs."""

        # Separate headings and content chunks
        headings = [l for l in ldus if l.chunk_type == ChunkType.heading]
        content_ldus = [l for l in ldus if l.chunk_type != ChunkType.heading]

        if not headings:
            # No section structure — create a single root node
            root = self._make_node(doc_id, "Document", 1, max(l.page_refs[-1] for l in ldus) if ldus else 1, 0, 0)
            root.chunk_ids = [l.chunk_id for l in ldus]
            root.data_types_present = self._detect_data_types(ldus)
            root.key_entities = self._extract_entities(ldus)
            if self._enable_summaries:
                root.summary = self._generate_summary(ldus[:10])
            else:
                root.summary = self._fallback_summary(ldus[:10])
            return PageIndex(
                doc_id=doc_id,
                doc_name=doc_name,
                root_nodes=[root],
                total_sections=1,
            )

        # Build tree from headings
        root_nodes: list[PageIndexNode] = []
        node_stack: list[PageIndexNode] = []  # stack for nesting
        all_nodes: list[PageIndexNode] = []
        heading_to_node: dict[str, PageIndexNode] = {}

        for i, heading in enumerate(headings):
            depth = heading.section_depth
            page_start = heading.page_refs[0] if heading.page_refs else 1

            # Determine page_end from next heading or document end
            if i + 1 < len(headings):
                next_page = headings[i + 1].page_refs[0] if headings[i + 1].page_refs else page_start
                page_end = max(page_start, next_page)
            else:
                page_end = max(l.page_refs[-1] for l in ldus) if ldus else page_start

            node = self._make_node(
                doc_id, heading.content.strip(),
                page_start, page_end, depth, len(all_nodes),
            )
            all_nodes.append(node)
            heading_to_node[heading.chunk_id] = node

            # Find parent by popping stack until we find a shallower depth
            while node_stack and node_stack[-1].depth >= depth:
                node_stack.pop()

            if node_stack:
                parent = node_stack[-1]
                node.parent_node_id = parent.node_id
                parent.child_nodes.append(node)
            else:
                root_nodes.append(node)

            node_stack.append(node)

        # Assign content LDUs to their section nodes
        self._assign_chunks_to_nodes(content_ldus, all_nodes, headings)

        # Detect data types and entities for each node
        ldu_map = {l.chunk_id: l for l in ldus}
        for node in all_nodes:
            node_ldus = [ldu_map[cid] for cid in node.chunk_ids if cid in ldu_map]
            node.data_types_present = self._detect_data_types(node_ldus)
            node.key_entities = self._extract_entities(node_ldus)
            if self._enable_summaries:
                node.summary = self._generate_summary(node_ldus[:8])
            else:
                node.summary = self._fallback_summary(node_ldus[:8])

        return PageIndex(
            doc_id=doc_id,
            doc_name=doc_name,
            root_nodes=root_nodes,
            total_sections=len(all_nodes),
        )

    def save(self, index: PageIndex, output_dir: str | Path = ".refinery/pageindex") -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{index.doc_id}.json"
        out_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
        logger.info("PageIndex saved -> %s", out_path)
        return out_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_node(
        self, doc_id: str, title: str,
        page_start: int, page_end: int, depth: int, seq: int,
    ) -> PageIndexNode:
        return PageIndexNode(
            node_id=f"{doc_id}_sec_{depth}_{seq:04d}",
            title=title,
            page_start=page_start,
            page_end=page_end,
            depth=depth,
        )

    def _assign_chunks_to_nodes(
        self, content_ldus: list[LDU],
        nodes: list[PageIndexNode],
        headings: list[LDU],
    ) -> None:
        """Assign each content LDU to its nearest preceding section node."""
        if not nodes:
            return

        # Build section boundaries: (section_title, node)
        section_map: dict[str, PageIndexNode] = {}
        for h, node in zip(headings, nodes):
            section_map[h.content.strip()] = node

        for ldu in content_ldus:
            if ldu.parent_section and ldu.parent_section in section_map:
                section_map[ldu.parent_section].chunk_ids.append(ldu.chunk_id)
            else:
                # Assign to the nearest node by page number
                best_node = nodes[0]
                for node in nodes:
                    if node.page_start <= (ldu.page_refs[0] if ldu.page_refs else 1):
                        best_node = node
                best_node.chunk_ids.append(ldu.chunk_id)

    def _detect_data_types(self, ldus: list[LDU]) -> list[str]:
        types = set()
        for ldu in ldus:
            if ldu.chunk_type == ChunkType.table:
                types.add("table")
            elif ldu.chunk_type == ChunkType.figure:
                types.add("figure")
            elif ldu.chunk_type == ChunkType.equation:
                types.add("equation")
            elif ldu.chunk_type == ChunkType.list_block:
                types.add("list")
            elif ldu.chunk_type == ChunkType.footnote:
                types.add("footnote")
        return sorted(types)

    def _extract_entities(self, ldus: list[LDU]) -> list[str]:
        """Simple entity extraction from chunk content."""
        text = " ".join(l.content[:200] for l in ldus[:10]).lower()
        entities = set()

        # Monetary amounts
        import re
        for match in re.findall(r'(?:birr|etb|usd|\$)\s*[\d,]+(?:\.\d+)?', text, re.IGNORECASE):
            entities.add(match.strip())
        for match in re.findall(r'[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand)', text, re.IGNORECASE):
            entities.add(match.strip())

        # Date patterns
        for match in re.findall(r'\b(?:20\d{2}[/-]?\d{0,2}|fy\s*\d{4})\b', text, re.IGNORECASE):
            entities.add(match.strip())

        # Named organisations (capitalised multi-word phrases)
        full_text = " ".join(l.content[:200] for l in ldus[:10])
        for match in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b', full_text):
            if len(match) > 5:
                entities.add(match)

        return sorted(entities)[:15]

    def _fallback_summary(self, ldus: list[LDU]) -> str:
        """Generate a summary without LLM — first 2-3 sentences of content."""
        text = " ".join(l.content for l in ldus if l.chunk_type == ChunkType.paragraph)
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        return ". ".join(sentences[:3]) + "." if sentences else "No summary available."

    def _generate_summary(self, ldus: list[LDU]) -> str:
        """Generate a 2-3 sentence section summary using a fast LLM."""
        content = "\n".join(l.content[:300] for l in ldus[:8])
        if not content.strip():
            return "No content available for summarisation."

        try:
            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Summarise the following document section in exactly 2-3 sentences. "
                                "Focus on key facts, figures, and topics.\n\n"
                                f"{content[:2000]}"
                            ),
                        }
                    ],
                    "max_tokens": 150,
                    "temperature": 0.1,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("LLM summary failed: %s — using fallback", exc)
            return self._fallback_summary(ldus)
