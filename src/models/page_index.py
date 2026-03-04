"""
PageIndex — the hierarchical navigation tree over a document.

Inspired by VectifyAI's PageIndex.  The tree mirrors the document's
section hierarchy and enables LLM-guided navigation: given a user query,
traverse the tree to identify the most relevant sections *before* doing
vector search, dramatically reducing the search space.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PageIndexNode(BaseModel):
    """
    A single node in the PageIndex tree — corresponds to one section.

    Child nodes represent sub-sections.  Leaf nodes may correspond to
    individual tables, figures, or densely structured paragraphs.
    """
    node_id: str = Field(description="Stable ID: {doc_id}_sec_{depth}_{sequence}")
    title: str = Field(description="Section heading text as it appears in the document")
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    depth: int = Field(ge=0, description="0 = root / document level, 1 = chapter, 2 = section …")

    # ── Content summary ───────────────────────────────────────────────────────
    summary: Optional[str] = Field(
        default=None,
        description="LLM-generated 2-3 sentence summary (populated by PageIndex builder)",
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="Named entities extracted from the section (organisations, dates, amounts …)",
    )
    data_types_present: list[str] = Field(
        default_factory=list,
        description="Content types in this section: table | figure | equation | list | footnote",
    )

    # ── Chunk references ──────────────────────────────────────────────────────
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="chunk_id values of all LDUs that belong to this section",
    )

    # ── Tree structure ────────────────────────────────────────────────────────
    parent_node_id: Optional[str] = None
    child_nodes: list["PageIndexNode"] = Field(default_factory=list)

    class Config:
        # Allow self-referential child_nodes
        arbitrary_types_allowed = True

    def to_flat_dict(self) -> dict:
        """Flattened representation (without child_nodes) for serialisation."""
        d = self.model_dump(exclude={"child_nodes"})
        return d


class PageIndex(BaseModel):
    """
    The complete navigation index for one document.

    Stored at  .refinery/pageindex/{doc_id}.json
    """
    doc_id: str
    doc_name: str
    root_nodes: list[PageIndexNode] = Field(
        description="Top-level section nodes (depth=0 or 1 depending on document structure)"
    )
    total_sections: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    index_version: str = Field(default="1.0.0")

    def find_section(self, node_id: str) -> Optional[PageIndexNode]:
        """BFS search for a node by ID."""
        queue = list(self.root_nodes)
        while queue:
            node = queue.pop(0)
            if node.node_id == node_id:
                return node
            queue.extend(node.child_nodes)
        return None

    def navigate(self, topic: str, top_k: int = 3) -> list[PageIndexNode]:
        """
        Keyword-based navigation: return top_k nodes whose title or
        key_entities contain the topic string (case-insensitive).

        Production implementation replaces this with embedding similarity.
        """
        topic_lower = topic.lower()
        scored: list[tuple[float, PageIndexNode]] = []

        def score_node(node: PageIndexNode) -> None:
            title_match = topic_lower in node.title.lower()
            entity_match = any(topic_lower in e.lower() for e in node.key_entities)
            summary_match = node.summary and topic_lower in node.summary.lower()
            score = title_match * 3 + entity_match * 2 + summary_match * 1
            if score > 0:
                scored.append((score, node))
            for child in node.child_nodes:
                score_node(child)

        for root in self.root_nodes:
            score_node(root)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
