"""RAG pipeline for medical knowledge retrieval and generation."""
from typing import Any

from loguru import logger

from src.config import get_settings
from src.core.exceptions import RAGError
from src.rag.vectorstore import DuckDBVectorStore


class RAGPipeline:
    """Complete RAG pipeline for medical knowledge retrieval.

    Implements:
    - Document chunking
    - Embedding generation
    - Similarity search
    - Context-aware generation
    """

    def __init__(self, vector_store: DuckDBVectorStore | None = None):
        """Initialize RAG pipeline.

        Args:
            vector_store: Optional vector store instance. Creates new if not provided.
        """
        self.settings = get_settings()
        self.vector_store = vector_store or DuckDBVectorStore()

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap

        # Simple chunking by character count
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > 0:
                    chunk = chunk[:break_point + 1]
                    end = start + len(chunk)

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]  # Remove empty chunks

    def ingest_document(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None
    ) -> dict[str, Any]:
        """Ingest a document into the RAG system.

        Args:
            text: Document text
            metadata: Optional metadata
            doc_id: Optional document ID

        Returns:
            Ingestion result with chunk count and IDs

        Raises:
            RAGError: If ingestion fails
        """
        try:
            logger.info(f"Ingesting document: {doc_id or 'unnamed'}")

            # Chunk the document
            chunks = self._chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks")

            # Prepare metadata for each chunk
            chunk_metadatas = []
            chunk_ids = []

            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_doc_id": doc_id or "unknown"
                })
                chunk_metadatas.append(chunk_metadata)

                chunk_id = f"{doc_id or 'doc'}_{i}" if doc_id else None
                chunk_ids.append(chunk_id)

            # Add to vector store
            db_ids = self.vector_store.add_documents(
                texts=chunks,
                metadatas=chunk_metadatas,
                doc_ids=chunk_ids
            )

            return {
                "success": True,
                "doc_id": doc_id,
                "chunks_created": len(chunks),
                "db_ids": db_ids
            }

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise RAGError(
                "Failed to ingest document",
                details={"error": str(e), "doc_id": doc_id}
            )

    def query(
        self,
        query: str,
        k: int | None = None,
        filter_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Query the RAG system for relevant information.

        Args:
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            Query results with retrieved documents and scores

        Raises:
            RAGError: If query fails
        """
        try:
            logger.info(f"Querying RAG system: {query[:100]}...")

            # Search for similar documents
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter_metadata=filter_metadata
            )

            # Extract and format context
            contexts = [r["content"] for r in results]

            return {
                "success": True,
                "query": query,
                "results": results,
                "contexts": contexts,
                "num_results": len(results)
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise RAGError(
                "Failed to query RAG system",
                details={"error": str(e), "query": query}
            )

    def generate_with_context(
        self,
        query: str,
        llm_caller: Any,
        k: int | None = None
    ) -> dict[str, Any]:
        """Generate response using retrieved context.

        Args:
            query: User query
            llm_caller: Function to call LLM with signature (prompt: str) -> str
            k: Number of context documents to retrieve

        Returns:
            Generated response with sources

        Raises:
            RAGError: If generation fails
        """
        try:
            # Retrieve relevant context
            query_result = self.query(query, k=k)
            contexts = query_result["contexts"]

            # Build prompt with context
            context_text = "\n\n".join([
                f"[Context {i+1}]: {ctx}"
                for i, ctx in enumerate(contexts)
            ])

            rag_prompt = f"""Based on the following medical information, answer the question.

MEDICAL KNOWLEDGE:
{context_text}

QUESTION: {query}

Provide a clear, accurate answer based on the provided context. If the context doesn't contain enough information, clearly state that."""

            # Generate response
            response = await llm_caller(rag_prompt)

            return {
                "success": True,
                "query": query,
                "response": response,
                "sources": query_result["results"],
                "num_sources": len(contexts)
            }

        except Exception as e:
            logger.error(f"Context-based generation failed: {e}")
            raise RAGError(
                "Failed to generate response with context",
                details={"error": str(e), "query": query}
            )

    def close(self) -> None:
        """Close RAG pipeline and cleanup resources."""
        self.vector_store.close()

    def __enter__(self) -> "RAGPipeline":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
