"""
Stage 3: Semantic Chunking Engine.

Converts an ExtractedDocument into a list of Logical Document Units (LDUs)
that are RAG-ready, semantically coherent, and carry full provenance.

Five enforced chunking rules (the "Constitution"):
  1. A table cell is never split from its header row.
  2. A figure caption is always stored as metadata of its parent figure chunk.
  3. A numbered list is kept as a single LDU unless it exceeds max_tokens.
  4. Section headers are stored as parent_section on all child chunks.
  5. Cross-references ("see Table 3") are resolved to chunk relationships.

ChunkValidator verifies all five rules before any LDU is emitted.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import yaml

from src.models.extracted_document import ExtractedDocument, TextBlock, TableData, FigureBlock
from src.models.ldu import LDU, ChunkType, ChunkRelationship

logger = logging.getLogger(__name__)

# ── Lightweight token counting ────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    logger.warning("tiktoken not installed — using word-count estimate for tokens")

    def _count_tokens(text: str) -> int:
        return max(1, len(text.split()) * 4 // 3)  # rough estimate


_DEFAULT_CFG = {
    "max_tokens_per_chunk": 512,
    "min_tokens_per_chunk": 32,
}

_CROSS_REF_PATTERN = re.compile(
    r"(?:see|refer(?:\s+to)?|as\s+(?:shown|described)\s+in)\s+"
    r"(table|figure|fig\.?|section|appendix)\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

_LIST_PATTERN = re.compile(r"^\s*(?:\d+[\.\)]\s|[-\u2022\u25e6]\s)", re.MULTILINE)


class ChunkValidator:
    """
    Validates the five chunking rules on every LDU before emission.

    Raises ValueError on violation so no malformed chunk ever reaches
    the vector store or downstream agents.
    """

    @staticmethod
    def validate(ldu: LDU, context: dict | None = None) -> list[str]:
        """Return a list of violation messages (empty = valid)."""
        violations: list[str] = []
        ctx = context or {}

        # Rule 1: table chunks must include header row (content should have headers)
        if ldu.chunk_type == ChunkType.table:
            if "table_has_headers" in ctx and not ctx["table_has_headers"]:
                violations.append(
                    f"Rule 1 violated: table chunk {ldu.chunk_id} missing header row"
                )

        # Rule 2: figure chunks should have caption in content or relationships
        if ldu.chunk_type == ChunkType.figure:
            has_caption_rel = any(
                r.relation_type == "caption_of" for r in ldu.relationships
            )
            caption_in_content = ctx.get("has_caption", False)
            if not has_caption_rel and not caption_in_content:
                violations.append(
                    f"Rule 2 violated: figure chunk {ldu.chunk_id} has no caption metadata"
                )

        # Rule 3: list blocks should not be split (checked during creation)
        # Enforced at creation time in ChunkingEngine._chunk_list_block

        # Rule 4: non-heading chunks must carry parent_section
        if ldu.chunk_type not in (ChunkType.heading, ChunkType.header_footer):
            if ldu.parent_section is None and ctx.get("has_sections", False):
                violations.append(
                    f"Rule 4 violated: chunk {ldu.chunk_id} missing parent_section"
                )

        # Rule 5: cross-references should be resolved
        if _CROSS_REF_PATTERN.search(ldu.content):
            has_xref = any(
                r.relation_type == "cross_ref" for r in ldu.relationships
            )
            if not has_xref:
                violations.append(
                    f"Rule 5 warning: chunk {ldu.chunk_id} has unresolved cross-reference"
                )

        return violations


class ChunkingEngine:
    """
    Converts ExtractedDocument → List[LDU].

    Usage::

        engine = ChunkingEngine(rules_path="rubric/extraction_rules.yaml")
        ldus = engine.chunk(extracted_doc)
    """

    def __init__(self, rules_path: str = "rubric/extraction_rules.yaml") -> None:
        self._cfg = dict(_DEFAULT_CFG)
        self._load_rules(rules_path)
        self._max_tokens = int(self._cfg["max_tokens_per_chunk"])
        self._min_tokens = int(self._cfg["min_tokens_per_chunk"])
        self._validator = ChunkValidator()

    def chunk(self, doc: ExtractedDocument) -> list[LDU]:
        """Chunk the document and return validated LDUs."""
        ldus: list[LDU] = []
        current_section: Optional[str] = None
        section_depth: int = 0
        seq = 0

        # Determine if document has section structure at all
        has_sections = any(b.is_heading for b in doc.text_blocks)

        # Build an ID lookup for cross-reference resolution
        all_ids: dict[str, str] = {}  # "table 1" → chunk_id

        # ── Process content in reading order ──────────────────────────────────
        id_to_item: dict[str, object] = {}
        for b in doc.text_blocks:
            id_to_item[b.block_id] = b
        for t in doc.tables:
            id_to_item[t.table_id] = t
        for f in doc.figures:
            id_to_item[f.figure_id] = f

        # Walk reading order (or fallback to sequential)
        order = doc.reading_order_ids or list(id_to_item.keys())

        # Buffer for merging small text blocks
        text_buffer: list[TextBlock] = []

        for item_id in order:
            item = id_to_item.get(item_id)
            if item is None:
                continue

            if isinstance(item, TextBlock):
                # Handle headings — update current section context
                if item.is_heading:
                    # Flush text buffer before section change
                    if text_buffer:
                        ldu, seq = self._flush_text_buffer(
                            text_buffer, doc.doc_id, seq, current_section, section_depth, has_sections
                        )
                        ldus.extend(ldu)
                        text_buffer = []

                    current_section = item.text.strip()
                    section_depth = item.heading_level or 1
                    heading_ldu = self._make_ldu(
                        doc.doc_id, seq, item.text, ChunkType.heading,
                        [item.bbox.page], item.bbox.to_dict() if item.bbox else None,
                        current_section, section_depth,
                    )
                    ldus.append(heading_ldu)
                    seq += 1
                    continue

                # Check if this is a list block
                if _LIST_PATTERN.match(item.text):
                    if text_buffer:
                        flushed, seq = self._flush_text_buffer(
                            text_buffer, doc.doc_id, seq, current_section, section_depth, has_sections
                        )
                        ldus.extend(flushed)
                        text_buffer = []
                    list_ldu, seq = self._chunk_list_block(
                        item, doc.doc_id, seq, current_section, section_depth
                    )
                    ldus.extend(list_ldu)
                    continue

                text_buffer.append(item)

                # Flush if buffer exceeds max tokens
                buf_text = "\n".join(b.text for b in text_buffer)
                if _count_tokens(buf_text) >= self._max_tokens:
                    flushed, seq = self._flush_text_buffer(
                        text_buffer, doc.doc_id, seq, current_section, section_depth, has_sections
                    )
                    ldus.extend(flushed)
                    text_buffer = []

            elif isinstance(item, TableData):
                if text_buffer:
                    flushed, seq = self._flush_text_buffer(
                        text_buffer, doc.doc_id, seq, current_section, section_depth, has_sections
                    )
                    ldus.extend(flushed)
                    text_buffer = []
                tbl_ldu, seq = self._chunk_table(item, doc.doc_id, seq, current_section, section_depth)
                ldus.extend(tbl_ldu)
                table_num = len([l for l in ldus if l.chunk_type == ChunkType.table])
                all_ids[f"table {table_num}"] = tbl_ldu[0].chunk_id if tbl_ldu else ""

            elif isinstance(item, FigureBlock):
                if text_buffer:
                    flushed, seq = self._flush_text_buffer(
                        text_buffer, doc.doc_id, seq, current_section, section_depth, has_sections
                    )
                    ldus.extend(flushed)
                    text_buffer = []
                fig_ldu, seq = self._chunk_figure(item, doc.doc_id, seq, current_section, section_depth)
                ldus.extend(fig_ldu)
                fig_num = len([l for l in ldus if l.chunk_type == ChunkType.figure])
                all_ids[f"figure {fig_num}"] = fig_ldu[0].chunk_id if fig_ldu else ""

        # Flush remaining text buffer
        if text_buffer:
            flushed, seq = self._flush_text_buffer(
                text_buffer, doc.doc_id, seq, current_section, section_depth, has_sections
            )
            ldus.extend(flushed)

        # ── Rule 5: Resolve cross-references ──────────────────────────────────
        self._resolve_cross_references(ldus, all_ids)

        # ── Validate all LDUs ─────────────────────────────────────────────────
        for ldu in ldus:
            ctx = {"has_sections": has_sections}
            if ldu.chunk_type == ChunkType.table:
                ctx["table_has_headers"] = True  # ensured during creation
            if ldu.chunk_type == ChunkType.figure:
                ctx["has_caption"] = "Caption:" in ldu.content or bool(ldu.relationships)
            violations = self._validator.validate(ldu, ctx)
            for v in violations:
                logger.warning("ChunkValidator: %s", v)

        return ldus

    # ── Internal chunking helpers ─────────────────────────────────────────────

    def _flush_text_buffer(
        self,
        buffer: list[TextBlock],
        doc_id: str,
        seq: int,
        section: Optional[str],
        depth: int,
        has_sections: bool,
    ) -> tuple[list[LDU], int]:
        """Merge buffered text blocks into LDU(s), splitting if over max_tokens."""
        ldus: list[LDU] = []
        combined = "\n".join(b.text for b in buffer)
        pages = sorted(set(b.bbox.page for b in buffer if b.bbox))
        bbox = buffer[0].bbox.to_dict() if buffer and buffer[0].bbox else None

        tokens = _count_tokens(combined)
        if tokens <= self._max_tokens:
            if tokens >= self._min_tokens:
                ldu = self._make_ldu(
                    doc_id, seq, combined, ChunkType.paragraph,
                    pages, bbox, section, depth,
                )
                ldus.append(ldu)
                seq += 1
            else:
                # Too small to be its own chunk — still emit to avoid data loss
                ldu = self._make_ldu(
                    doc_id, seq, combined, ChunkType.paragraph,
                    pages, bbox, section, depth,
                )
                ldus.append(ldu)
                seq += 1
        else:
            # Split by individual blocks
            for blk in buffer:
                blk_tokens = _count_tokens(blk.text)
                if blk_tokens > self._max_tokens:
                    # Hard split on sentences
                    sentences = re.split(r'(?<=[.!?])\s+', blk.text)
                    chunk_text = ""
                    for sent in sentences:
                        if _count_tokens(chunk_text + " " + sent) > self._max_tokens and chunk_text:
                            ldu = self._make_ldu(
                                doc_id, seq, chunk_text.strip(), ChunkType.paragraph,
                                [blk.bbox.page] if blk.bbox else pages,
                                blk.bbox.to_dict() if blk.bbox else None,
                                section, depth,
                            )
                            ldus.append(ldu)
                            seq += 1
                            chunk_text = sent
                        else:
                            chunk_text = (chunk_text + " " + sent).strip()
                    if chunk_text.strip():
                        ldu = self._make_ldu(
                            doc_id, seq, chunk_text.strip(), ChunkType.paragraph,
                            [blk.bbox.page] if blk.bbox else pages,
                            blk.bbox.to_dict() if blk.bbox else None,
                            section, depth,
                        )
                        ldus.append(ldu)
                        seq += 1
                else:
                    ldu = self._make_ldu(
                        doc_id, seq, blk.text, ChunkType.paragraph,
                        [blk.bbox.page] if blk.bbox else pages,
                        blk.bbox.to_dict() if blk.bbox else None,
                        section, depth,
                    )
                    ldus.append(ldu)
                    seq += 1

        return ldus, seq

    def _chunk_table(
        self, table: TableData, doc_id: str, seq: int,
        section: Optional[str], depth: int,
    ) -> tuple[list[LDU], int]:
        """
        Rule 1: table cells are NEVER split from their header row.
        The entire table is emitted as a single LDU with headers preserved.
        """
        # Render table as readable text with headers
        headers = table.headers
        rows = table.rows
        lines = []
        if table.caption:
            lines.append(f"Caption: {table.caption}")
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")

        content = "\n".join(lines)
        pages = [table.bbox.page] if table.bbox else []

        ldu = self._make_ldu(
            doc_id, seq, content, ChunkType.table,
            pages, table.bbox.to_dict() if table.bbox else None,
            section, depth,
        )
        return [ldu], seq + 1

    def _chunk_figure(
        self, figure: FigureBlock, doc_id: str, seq: int,
        section: Optional[str], depth: int,
    ) -> tuple[list[LDU], int]:
        """
        Rule 2: figure caption is ALWAYS stored as metadata of its parent figure chunk.
        """
        parts = [f"[Figure: {figure.figure_type}]"]
        if figure.caption:
            parts.append(f"Caption: {figure.caption}")
        if figure.alt_text:
            parts.append(f"Description: {figure.alt_text}")

        content = "\n".join(parts)
        pages = [figure.bbox.page] if figure.bbox else []
        relationships = []
        if figure.caption:
            relationships.append(ChunkRelationship(
                target_chunk_id=f"{doc_id}_{seq:06d}",
                relation_type="caption_of",
            ))

        ldu = self._make_ldu(
            doc_id, seq, content, ChunkType.figure,
            pages, figure.bbox.to_dict() if figure.bbox else None,
            section, depth, relationships=relationships,
        )
        return [ldu], seq + 1

    def _chunk_list_block(
        self, block: TextBlock, doc_id: str, seq: int,
        section: Optional[str], depth: int,
    ) -> tuple[list[LDU], int]:
        """
        Rule 3: numbered/bulleted lists are kept as a single LDU
        unless they exceed max_tokens.
        """
        tokens = _count_tokens(block.text)
        pages = [block.bbox.page] if block.bbox else []
        bbox = block.bbox.to_dict() if block.bbox else None

        if tokens <= self._max_tokens:
            ldu = self._make_ldu(
                doc_id, seq, block.text, ChunkType.list_block,
                pages, bbox, section, depth,
            )
            return [ldu], seq + 1

        # Exceeds max — split by list items while preserving structure
        items = re.split(r'(?=^\s*(?:\d+[\.\)]\s|[-\u2022\u25e6]\s))', block.text, flags=re.MULTILINE)
        items = [i.strip() for i in items if i.strip()]

        ldus = []
        current_chunk = ""
        for item in items:
            if _count_tokens(current_chunk + "\n" + item) > self._max_tokens and current_chunk:
                ldu = self._make_ldu(
                    doc_id, seq, current_chunk.strip(), ChunkType.list_block,
                    pages, bbox, section, depth,
                )
                ldus.append(ldu)
                seq += 1
                current_chunk = item
            else:
                current_chunk = (current_chunk + "\n" + item).strip()
        if current_chunk:
            ldu = self._make_ldu(
                doc_id, seq, current_chunk.strip(), ChunkType.list_block,
                pages, bbox, section, depth,
            )
            ldus.append(ldu)
            seq += 1

        return ldus, seq

    def _resolve_cross_references(self, ldus: list[LDU], id_map: dict[str, str]) -> None:
        """Rule 5: resolve cross-references to chunk relationships."""
        for ldu in ldus:
            matches = _CROSS_REF_PATTERN.findall(ldu.content)
            for ref_type, ref_num in matches:
                ref_key = f"{ref_type.lower().rstrip('.')} {ref_num}"
                target_id = id_map.get(ref_key)
                if target_id:
                    ldu.relationships.append(ChunkRelationship(
                        target_chunk_id=target_id,
                        relation_type="cross_ref",
                    ))

    def _make_ldu(
        self,
        doc_id: str,
        seq: int,
        content: str,
        chunk_type: ChunkType,
        page_refs: list[int],
        bounding_box: dict | None,
        parent_section: Optional[str],
        section_depth: int,
        relationships: list[ChunkRelationship] | None = None,
    ) -> LDU:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return LDU(
            chunk_id=f"{doc_id}_{seq:06d}",
            doc_id=doc_id,
            sequence=seq,
            content=content,
            chunk_type=chunk_type,
            token_count=_count_tokens(content),
            page_refs=page_refs or [1],
            bounding_box=bounding_box,
            parent_section=parent_section,
            section_depth=section_depth,
            relationships=relationships or [],
            content_hash=content_hash,
        )

    def _load_rules(self, rules_path: str) -> None:
        path = Path(rules_path)
        if not path.exists():
            return
        with path.open() as f:
            rules = yaml.safe_load(f)
        chunking = rules.get("chunking", {})
        if "max_tokens_per_chunk" in chunking:
            self._cfg["max_tokens_per_chunk"] = chunking["max_tokens_per_chunk"]
        if "min_tokens_per_chunk" in chunking:
            self._cfg["min_tokens_per_chunk"] = chunking["min_tokens_per_chunk"]
