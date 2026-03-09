"""
Vector Store Manager — ChromaDB / FAISS ingestion of LDUs.

Ingests Logical Document Units into a persistent vector store for
semantic search retrieval.  Used by the Query Interface Agent's
semantic_search tool.

Backend is configurable via extraction_rules.yaml (default: ChromaDB).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from src.models.ldu import LDU

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False
    logger.warning("ChromaDB not installed — vector store unavailable")

_DEFAULT_CFG = {
    "backend": "chromadb",
    "collection_name": "document_intelligence_refinery",
    "persist_directory": ".refinery/vectorstore",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
}


class VectorStoreManager:
    """
    Manages LDU ingestion and semantic search over a ChromaDB collection.

    Usage::

        store = VectorStoreManager()
        store.ingest(ldus)
        results = store.search("capital expenditure Q3", top_k=5)
    """

    def __init__(self, rules_path: str = "rubric/extraction_rules.yaml") -> None:
        self._cfg = dict(_DEFAULT_CFG)
        self._load_config(rules_path)
        self._collection = None
        self._client = None
        self._init_store()

    def _load_config(self, rules_path: str) -> None:
        path = Path(rules_path)
        if not path.exists():
            return
        with path.open() as f:
            rules = yaml.safe_load(f)
        vs_cfg = rules.get("vector_store", {})
        self._cfg.update(vs_cfg)

    def _init_store(self) -> None:
        if not _CHROMA_OK:
            logger.warning("ChromaDB not available — vector store operations will be no-ops")
            return

        persist_dir = Path(self._cfg["persist_directory"])
        persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self._cfg["collection_name"],
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready (%d documents)",
                self._cfg["collection_name"],
                self._collection.count(),
            )
        except Exception as exc:
            logger.error("ChromaDB initialization failed: %s", exc)

    def ingest(self, ldus: list[LDU], batch_size: int = 100) -> int:
        """
        Ingest LDUs into the vector store.
        Returns the number of documents ingested.
        """
        if not self._collection:
            logger.warning("No vector store collection — skipping ingestion")
            return 0

        count = 0
        for i in range(0, len(ldus), batch_size):
            batch = ldus[i : i + batch_size]
            ids = [ldu.chunk_id for ldu in batch]
            documents = [ldu.content for ldu in batch]
            metadatas = [
                {
                    "doc_id": ldu.doc_id,
                    "chunk_type": ldu.chunk_type if isinstance(ldu.chunk_type, str) else ldu.chunk_type,
                    "page_refs": ",".join(str(p) for p in ldu.page_refs),
                    "parent_section": ldu.parent_section or "",
                    "token_count": ldu.token_count,
                    "content_hash": ldu.content_hash,
                }
                for ldu in batch
            ]

            try:
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                count += len(batch)
            except Exception as exc:
                logger.error("Vector store ingestion batch error: %s", exc)

        logger.info("Ingested %d LDUs into vector store", count)
        return count

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search over ingested LDUs.

        Returns list of dicts with keys: chunk_id, content, metadata, distance.
        """
        if not self._collection:
            logger.warning("No vector store collection — returning empty results")
            return []

        where_filter = {"doc_id": doc_id} if doc_id else None

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
        except Exception as exc:
            logger.error("Vector store search error: %s", exc)
            return []

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                hits.append({
                    "chunk_id": chunk_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })

        return hits

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks for a document from the store."""
        if not self._collection:
            return 0
        try:
            existing = self._collection.get(where={"doc_id": doc_id})
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                return len(existing["ids"])
        except Exception as exc:
            logger.error("Vector store delete error: %s", exc)
        return 0

    @property
    def count(self) -> int:
        if self._collection:
            return self._collection.count()
        return 0
