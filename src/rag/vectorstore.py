"""DuckDB vector store implementation for embeddings."""
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.core.exceptions import RAGError


class DuckDBVectorStore:
    """Vector store using DuckDB for medical document embeddings.

    Features:
    - Efficient vector similarity search
    - Metadata storage
    - Automatic indexing
    - Batch operations support
    """

    def __init__(self, db_path: str | Path | None = None):
        """Initialize DuckDB vector store.

        Args:
            db_path: Path to DuckDB database file. Uses config default if not provided.
        """
        self.settings = get_settings()
        self.db_path = db_path or self.settings.vector_db_full_path

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.settings.embedding_model}")
        self.embedding_model = SentenceTransformer(self.settings.embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Connect to database
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create database schema for vector storage."""
        logger.info("Initializing DuckDB vector store schema")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                doc_id VARCHAR,
                content TEXT,
                metadata VARCHAR,
                embedding FLOAT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for faster similarity search
        try:
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)
            """)
        except Exception as e:
            logger.warning(f"Index creation failed (may already exist): {e}")

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        doc_ids: list[str] | None = None
    ) -> list[int]:
        """Add documents to vector store with embeddings.

        Args:
            texts: List of document texts to add
            metadatas: Optional metadata for each document
            doc_ids: Optional document IDs

        Returns:
            List of database row IDs

        Raises:
            RAGError: If document addition fails
        """
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

            # Prepare metadata
            if metadatas is None:
                metadatas = [{}] * len(texts)
            if doc_ids is None:
                doc_ids = [f"doc_{i}" for i in range(len(texts))]

            # Insert into database
            inserted_ids = []
            for i, (text, metadata, doc_id, embedding) in enumerate(
                zip(texts, metadatas, doc_ids, embeddings)
            ):
                result = self.conn.execute("""
                    INSERT INTO documents (doc_id, content, metadata, embedding)
                    VALUES (?, ?, ?, ?)
                    RETURNING id
                """, (doc_id, text, json.dumps(metadata), embedding.tolist()))

                inserted_ids.append(result.fetchone()[0])

            self.conn.commit()
            logger.info(f"Successfully added {len(inserted_ids)} documents")
            return inserted_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise RAGError(
                "Failed to add documents to vector store",
                details={"error": str(e)}
            )

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter_metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Search for similar documents using cosine similarity.

        Args:
            query: Query text
            k: Number of results to return. Uses config default if not provided.
            filter_metadata: Optional metadata filters

        Returns:
            List of matching documents with scores and metadata

        Raises:
            RAGError: If search fails
        """
        try:
            k = k or self.settings.top_k_results

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]

            # Calculate cosine similarity and retrieve top-k
            # Using array dot product for cosine similarity
            query_sql = """
                SELECT
                    id,
                    doc_id,
                    content,
                    metadata,
                    list_cosine_similarity(embedding, ?::FLOAT[]) as similarity_score
                FROM documents
            """

            params = [query_embedding.tolist()]

            # Add metadata filtering if provided
            if filter_metadata:
                # Simple metadata filtering (can be enhanced)
                query_sql += " WHERE metadata LIKE ?"
                params.append(f"%{list(filter_metadata.values())[0]}%")

            query_sql += """
                ORDER BY similarity_score DESC
                LIMIT ?
            """
            params.append(k)

            results = self.conn.execute(query_sql, params).fetchall()

            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "id": row[0],
                    "doc_id": row[1],
                    "content": row[2],
                    "metadata": json.loads(row[3]),
                    "similarity_score": float(row[4])
                })

            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RAGError(
                "Failed to perform similarity search",
                details={"error": str(e), "query": query}
            )

    def delete_documents(self, doc_ids: list[str]) -> int:
        """Delete documents by ID.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        try:
            result = self.conn.execute("""
                DELETE FROM documents
                WHERE doc_id IN ({})
            """.format(','.join(['?'] * len(doc_ids))), doc_ids)

            deleted_count = result.rowcount
            self.conn.commit()

            logger.info(f"Deleted {deleted_count} documents")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise RAGError(
                "Failed to delete documents",
                details={"error": str(e)}
            )

    def get_document_count(self) -> int:
        """Get total number of documents in store."""
        result = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return result[0]

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        logger.info("Vector store connection closed")

    def __enter__(self) -> "DuckDBVectorStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
