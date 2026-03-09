"""
Stage 5: Query Interface Agent — LangGraph agent with three tools.

Tools:
  1. pageindex_navigate — tree traversal to locate relevant sections
  2. semantic_search    — embedding-based vector retrieval (ChromaDB)
  3. structured_query   — SQL over the FactTable (SQLite)

Every answer includes a ProvenanceChain with source citations.

The agent orchestrates a two-phase retrieval:
  Phase 1: PageIndex navigation → narrow the search space
  Phase 2: Semantic search / structured query within the narrowed scope

This significantly outperforms naive full-corpus vector search for
section-specific queries (e.g. "What are the Q3 expenditure projections?").
"""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any, Optional, TypedDict

from src.models.page_index import PageIndex
from src.models.provenance import ProvenanceChain
from src.agents.auditor import AuditEngine
from src.storage.fact_table import FactTableDB
from src.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# ── Optional LangGraph / LangChain imports ────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    _LANGGRAPH_OK = True
except ImportError:
    _LANGGRAPH_OK = False
    logger.warning("LangGraph/LangChain not installed — QueryAgent uses fallback mode")

try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    doc_id: Optional[str]
    page_index_results: list[dict]
    semantic_results: list[dict]
    structured_results: list[dict]
    answer: str
    provenance: Optional[dict]
    tool_calls: list[str]


# ── Tool implementations ─────────────────────────────────────────────────────

class QueryTools:
    """The three retrieval tools available to the Query Agent."""

    def __init__(
        self,
        page_indexes: dict[str, PageIndex],
        vector_store: Optional[VectorStoreManager] = None,
        fact_db: Optional[FactTableDB] = None,
    ) -> None:
        self._indexes = page_indexes
        self._vs = vector_store
        self._db = fact_db

    def pageindex_navigate(self, query: str, doc_id: Optional[str] = None) -> list[dict]:
        """
        Tool 1: Navigate the PageIndex tree to find relevant sections.

        Returns top-3 sections matching the query topic, with their
        page ranges and chunk IDs for targeted retrieval.
        """
        results = []
        indexes_to_search = (
            {doc_id: self._indexes[doc_id]} if doc_id and doc_id in self._indexes
            else self._indexes
        )

        for did, index in indexes_to_search.items():
            nodes = index.navigate(query, top_k=3)
            for node in nodes:
                results.append({
                    "doc_id": did,
                    "doc_name": index.doc_name,
                    "section_title": node.title,
                    "page_start": node.page_start,
                    "page_end": node.page_end,
                    "summary": node.summary or "",
                    "key_entities": node.key_entities,
                    "data_types": node.data_types_present,
                    "chunk_ids": node.chunk_ids[:10],
                })

        logger.info("pageindex_navigate found %d sections for '%s'", len(results), query[:50])
        return results[:5]

    def semantic_search(self, query: str, doc_id: Optional[str] = None, top_k: int = 5) -> list[dict]:
        """
        Tool 2: Semantic search over the vector store.

        Returns top-k chunks ranked by embedding similarity.
        """
        if not self._vs:
            return []
        results = self._vs.search(query, top_k=top_k, doc_id=doc_id)
        logger.info("semantic_search found %d results for '%s'", len(results), query[:50])
        return results

    def structured_query(self, query: str, doc_id: Optional[str] = None) -> list[dict]:
        """
        Tool 3: SQL query over the FactTable.

        Translates natural language into a keyword search over extracted facts.
        For direct SQL, the query can start with "SQL:" prefix.
        """
        if not self._db:
            return []

        # Direct SQL mode
        if query.upper().startswith("SQL:"):
            sql = query[4:].strip()
            if not sql.upper().startswith("SELECT"):
                return [{"error": "Only SELECT queries are allowed"}]
            return self._db.query(sql)

        # Keyword extraction mode
        import re
        stopwords = {"what", "is", "the", "of", "in", "for", "and", "a", "an", "to", "how", "much", "were", "was"}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stopwords]

        all_results = []
        for kw in keywords[:3]:
            facts = self._db.search_facts(kw, doc_id=doc_id)
            all_results.extend(facts)

        # Deduplicate by id
        seen = set()
        unique = []
        for r in all_results:
            rid = r.get("id", id(r))
            if rid not in seen:
                seen.add(rid)
                unique.append(r)

        logger.info("structured_query found %d facts for '%s'", len(unique), query[:50])
        return unique[:10]


# ── Query Agent ───────────────────────────────────────────────────────────────

class QueryAgent:
    """
    LangGraph-based query agent with three retrieval tools.

    Usage::

        agent = QueryAgent(
            page_indexes={"doc_id": page_index},
            vector_store=vs_manager,
            fact_db=fact_db,
        )
        result = agent.query("What was the total revenue in FY 2023?", doc_id="abc")
        print(result["answer"])
        print(result["provenance"])
    """

    def __init__(
        self,
        page_indexes: dict[str, PageIndex] | None = None,
        vector_store: Optional[VectorStoreManager] = None,
        fact_db: Optional[FactTableDB] = None,
        api_key: Optional[str] = None,
        model: str = "google/gemini-flash-1.5",
    ) -> None:
        self._tools = QueryTools(
            page_indexes=page_indexes or {},
            vector_store=vector_store,
            fact_db=fact_db,
        )
        self._auditor = AuditEngine(
            vector_store=vector_store,
            fact_db=fact_db,
        )
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._model = model
        self._graph = self._build_graph() if _LANGGRAPH_OK else None

    def query(self, question: str, doc_id: Optional[str] = None) -> dict:
        """
        Answer a question using the three-tool retrieval pipeline.

        Returns dict with keys: answer, provenance, tool_calls.
        """
        if self._graph:
            return self._run_graph(question, doc_id)
        return self._run_fallback(question, doc_id)

    def verify_claim(self, claim: str, doc_id: Optional[str] = None) -> ProvenanceChain:
        """Audit Mode: verify a claim against stored evidence."""
        return self._auditor.verify_claim(claim, doc_id)

    # ── LangGraph implementation ──────────────────────────────────────────────

    def _build_graph(self):
        """Build the LangGraph state graph."""
        graph = StateGraph(AgentState)

        graph.add_node("navigate", self._node_navigate)
        graph.add_node("search", self._node_search)
        graph.add_node("synthesize", self._node_synthesize)

        graph.set_entry_point("navigate")
        graph.add_edge("navigate", "search")
        graph.add_edge("search", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _node_navigate(self, state: AgentState) -> dict:
        """Phase 1: PageIndex navigation."""
        results = self._tools.pageindex_navigate(state["query"], state.get("doc_id"))
        return {
            "page_index_results": results,
            "tool_calls": state.get("tool_calls", []) + ["pageindex_navigate"],
        }

    def _node_search(self, state: AgentState) -> dict:
        """Phase 2: Semantic search + structured query."""
        doc_id = state.get("doc_id")

        # Semantic search
        semantic_results = self._tools.semantic_search(state["query"], doc_id=doc_id)

        # Structured query
        structured_results = self._tools.structured_query(state["query"], doc_id=doc_id)

        return {
            "semantic_results": semantic_results,
            "structured_results": structured_results,
            "tool_calls": state.get("tool_calls", []) + ["semantic_search", "structured_query"],
        }

    def _node_synthesize(self, state: AgentState) -> dict:
        """Synthesize answer from all retrieved evidence."""
        context = self._build_context(state)
        answer = self._generate_answer(state["query"], context)

        # Build provenance chain
        supporting_chunks = state.get("semantic_results", [])
        provenance = self._auditor.build_provenance_for_answer(
            query=state["query"],
            answer=answer,
            supporting_chunks=supporting_chunks,
        )

        return {
            "answer": answer,
            "provenance": provenance.model_dump(),
        }

    def _run_graph(self, question: str, doc_id: Optional[str]) -> dict:
        """Execute the LangGraph pipeline."""
        initial_state: AgentState = {
            "query": question,
            "doc_id": doc_id,
            "page_index_results": [],
            "semantic_results": [],
            "structured_results": [],
            "answer": "",
            "provenance": None,
            "tool_calls": [],
        }
        result = self._graph.invoke(initial_state)
        return {
            "answer": result.get("answer", ""),
            "provenance": result.get("provenance"),
            "tool_calls": result.get("tool_calls", []),
        }

    # ── Fallback (no LangGraph) ───────────────────────────────────────────────

    def _run_fallback(self, question: str, doc_id: Optional[str]) -> dict:
        """Run the pipeline without LangGraph."""
        tool_calls = []

        # Phase 1: Navigate
        pi_results = self._tools.pageindex_navigate(question, doc_id)
        tool_calls.append("pageindex_navigate")

        # Phase 2: Search
        semantic_results = self._tools.semantic_search(question, doc_id=doc_id)
        tool_calls.append("semantic_search")

        structured_results = self._tools.structured_query(question, doc_id=doc_id)
        tool_calls.append("structured_query")

        # Synthesize
        context = self._build_context_from_results(pi_results, semantic_results, structured_results)
        answer = self._generate_answer(question, context)

        provenance = self._auditor.build_provenance_for_answer(
            query=question,
            answer=answer,
            supporting_chunks=semantic_results,
        )

        return {
            "answer": answer,
            "provenance": provenance.model_dump(),
            "tool_calls": tool_calls,
        }

    # ── Context building ──────────────────────────────────────────────────────

    def _build_context(self, state: AgentState) -> str:
        return self._build_context_from_results(
            state.get("page_index_results", []),
            state.get("semantic_results", []),
            state.get("structured_results", []),
        )

    def _build_context_from_results(
        self,
        pi_results: list[dict],
        semantic_results: list[dict],
        structured_results: list[dict],
    ) -> str:
        parts = []

        if pi_results:
            parts.append("=== Relevant Sections (PageIndex) ===")
            for r in pi_results[:3]:
                parts.append(
                    f"- {r.get('section_title', 'Untitled')} "
                    f"(pp. {r.get('page_start')}-{r.get('page_end')}): "
                    f"{r.get('summary', '')}"
                )

        if semantic_results:
            parts.append("\n=== Semantic Search Results ===")
            for r in semantic_results[:5]:
                content = r.get("content", "")[:300]
                meta = r.get("metadata", {})
                parts.append(
                    f"[p.{meta.get('page_refs', '?')}, {meta.get('parent_section', '')}] "
                    f"{content}"
                )

        if structured_results:
            parts.append("\n=== Structured Facts ===")
            for r in structured_results[:5]:
                parts.append(f"- {r.get('key', '')}: {r.get('value', '')} (p.{r.get('page_number', '?')})")

        return "\n".join(parts) if parts else "No relevant information found."

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the LLM or fallback to context summary."""
        if self._api_key and _HTTPX_OK:
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
                                "role": "system",
                                "content": (
                                    "You are a document analysis assistant. Answer the question "
                                    "based ONLY on the provided context. If the answer is not in "
                                    "the context, say 'The information is not available in the "
                                    "provided documents.' Always cite page numbers."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"Context:\n{context}\n\nQuestion: {question}",
                            },
                        ],
                        "max_tokens": 500,
                        "temperature": 0.1,
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.warning("LLM answer generation failed: %s — using context summary", exc)

        # Fallback: return context as the answer
        if context and context != "No relevant information found.":
            return f"Based on the retrieved evidence:\n\n{context[:500]}"
        return "The information is not available in the provided documents."
