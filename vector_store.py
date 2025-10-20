#!/usr/bin/env python3
"""
Vector Store Module - PostgreSQL + pgvector
Manages embedding storage, retrieval, and similarity search
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from loguru import logger
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pgvector.sqlalchemy import Vector
import numpy as np

# Configure logging
logger.remove()
logger.add("vector_store.log", format="<level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>", level="DEBUG")
logger.add(lambda msg: None)  # Silent stderr


Base = declarative_base()


class PaperEmbedding(Base):
    """SQLAlchemy model for paper embeddings"""

    __tablename__ = "paper_embeddings"

    id = Column(String, primary_key=True)
    paper_id = Column(String, nullable=False, index=True)
    chunk_type = Column(String, nullable=False)  # text, formula, table, theorem
    content = Column(Text)
    latex_content = Column(Text)
    embedding = Column(Vector(1024))  # Ensemble embedding dimension
    chunk_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PaperEmbedding(id={self.id}, type={self.chunk_type})>"


class VectorStore:
    """PostgreSQL + pgvector store for embeddings"""

    def __init__(
        self,
        db_url: str = None,
        create_tables: bool = True,
        create_index: bool = False,
        index_lists: int = None,
        echo: bool = False
    ):
        """
        Initialize vector store

        Args:
            db_url: PostgreSQL connection string
                   Format: postgresql://user:password@localhost:5432/dbname
            create_tables: Whether to create tables if they don't exist
            create_index: Whether to create IVFFlat index (default False)
                         For large datasets (40GB+), index creation can take hours
                         and requires significant memory. Only enable after all data is loaded.
            index_lists: Number of lists for IVFFlat index (default: auto-calculated)
                        Recommended: sqrt(num_vectors) to num_vectors/10
                        For 1M vectors: 100-1000 lists
                        For 10M vectors: 1000-3000 lists
            echo: SQLAlchemy echo for debugging
        """
        self.create_index = create_index
        self.index_lists = index_lists
        if db_url is None:
            # Try environment variable
            db_url = os.getenv(
                "DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/embed_rag"
            )

        logger.info(f"Connecting to: {db_url.split('@')[1] if '@' in db_url else 'local'}")

        try:
            self.engine = create_engine(db_url, echo=echo)
            self.Session = sessionmaker(bind=self.engine)

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(sa.text("SELECT 1"))
                logger.info("✓ Database connection successful")

            if create_tables:
                Base.metadata.create_all(self.engine)
                logger.info("✓ Tables created/verified")

                # Create vector index only if explicitly requested
                if self.create_index:
                    self._create_vector_index()
                else:
                    logger.info("⚠️  Index creation disabled (create_index=False)")
                    logger.info("   For large datasets, create index AFTER loading all data")
                    logger.info("   Use: store.create_index_now(lists=1000)")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _create_vector_index(self):
        """Internal method - create index with auto-calculated lists"""
        with self.engine.connect() as conn:
            try:
                # Check how many embeddings we have
                result = conn.execute(sa.text(
                    "SELECT COUNT(*) FROM paper_embeddings"
                )).scalar()

                embedding_count = result or 0

                if embedding_count < 100:
                    logger.warning(f"Only {embedding_count} embeddings - IVFFlat needs >1000 for good recall")
                    logger.warning("Proceeding anyway since create_index=True was set")

                # Auto-calculate lists if not specified
                if self.index_lists is None:
                    # Heuristic: sqrt(N) to N/10
                    # For 1M vectors: ~1000 lists
                    # For 10M vectors: ~3000 lists
                    import math
                    lists = min(3000, max(100, int(math.sqrt(embedding_count))))
                else:
                    lists = self.index_lists

                logger.info(f"Creating IVFFlat index with {lists} lists for {embedding_count} vectors...")

                # Drop old index if exists
                conn.execute(sa.text("DROP INDEX IF EXISTS ix_embedding"))
                conn.commit()

                # Create new index
                conn.execute(sa.text(
                    f"CREATE INDEX ix_embedding ON paper_embeddings "
                    f"USING ivfflat (embedding vector_cosine_ops) WITH (lists={lists})"
                ))
                conn.commit()
                logger.info(f"✓ IVFFlat index created successfully")

            except Exception as e:
                logger.error(f"Index creation failed: {e}")
                raise

    def create_index_now(self, lists: int = None, force: bool = False):
        """
        Create vector index manually (after loading all data)

        Args:
            lists: Number of lists for IVFFlat (default: auto-calculated)
                  Recommended ranges:
                  - 100K-1M vectors: 500-1000 lists
                  - 1M-10M vectors: 1000-3000 lists
                  - 10M+ vectors: 3000-5000 lists
            force: Create index even if < 1000 vectors (not recommended)

        Example:
            # After loading 5M vectors
            store.create_index_now(lists=2000)
        """
        session = self.Session()
        try:
            count = session.query(sa.func.count(PaperEmbedding.id)).scalar()

            if count < 1000 and not force:
                raise ValueError(
                    f"Only {count} vectors in database. IVFFlat index needs >1000 for good recall.\n"
                    f"Use force=True to create anyway (not recommended for production)"
                )

            logger.info(f"Creating index for {count:,} vectors...")

            # Temporarily set parameters
            old_lists = self.index_lists
            self.index_lists = lists

            self._create_vector_index()

            # Restore
            self.index_lists = old_lists

            logger.info("✓ Index creation complete")

        finally:
            session.close()

    def insert_embedding(
        self,
        chunk_id: str,
        paper_id: str,
        chunk_type: str,
        content: str,
        embedding: List[float],
        metadata: Dict = None,
        latex_content: str = None
    ) -> bool:
        """Insert single embedding"""
        session = self.Session()
        try:
            record = PaperEmbedding(
                id=chunk_id,
                paper_id=paper_id,
                chunk_type=chunk_type,
                content=content,
                latex_content=latex_content,
                embedding=embedding,
                chunk_metadata=metadata or {}
            )
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def insert_batch(self, records: List[Dict]) -> int:
        """
        Insert batch of embeddings

        Args:
            records: List of {
                "id": str,
                "paper_id": str,
                "chunk_type": str,
                "content": str,
                "embedding": List[float],
                "metadata": Dict (optional),
                "latex_content": str (optional)
            }

        Returns:
            Number of records inserted
        """
        session = self.Session()
        inserted = 0

        try:
            for record in records:
                paper_emb = PaperEmbedding(
                    id=record["id"],
                    paper_id=record["paper_id"],
                    chunk_type=record["chunk_type"],
                    content=record.get("content", ""),
                    latex_content=record.get("latex_content"),
                    embedding=record["embedding"],
                    chunk_metadata=record.get("metadata", {})
                )
                session.add(paper_emb)
                inserted += 1

            session.commit()
            logger.info(f"✓ Inserted {inserted} records")
            return inserted

        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            session.rollback()
            return inserted
        finally:
            session.close()

    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        chunk_type: str = None,
        paper_id: str = None,
        threshold: float = 0.0,
        probes: int = 10
    ) -> List[Dict]:
        """
        Search similar embeddings using cosine similarity

        Args:
            query_embedding: Query vector (1024-dim)
            limit: Number of results to return
            chunk_type: Filter by chunk type (optional)
            paper_id: Filter by paper (optional)
            threshold: Similarity threshold (0-1, default 0.0 for all results)
            probes: Number of IVFFlat probes (higher = better recall, slower)
                   Default 10 for good balance. Range: 1-100

        Returns:
            List of similar records with scores
        """
        session = self.Session()

        try:
            # Set IVFFlat probes for better recall
            # Higher probes = more lists scanned = better results but slower
            session.execute(sa.text(f"SET ivfflat.probes = {probes}"))

            # Build query
            query = session.query(
                PaperEmbedding,
                PaperEmbedding.embedding.cosine_distance(query_embedding).label("distance")
            )

            # Add filters
            if chunk_type:
                query = query.filter(PaperEmbedding.chunk_type == chunk_type)
            if paper_id:
                query = query.filter(PaperEmbedding.paper_id == paper_id)

            # Order by similarity (ascending distance = highest similarity)
            results = query.order_by("distance").limit(limit * 2).all()  # Get more to debug

            logger.debug(f"Vector search returned {len(results)} raw results")

            # Format results
            formatted = []
            for i, (record, distance) in enumerate(results):
                similarity = 1 - distance  # Convert distance to similarity

                if i < 3:  # Log first 3 for debugging
                    logger.debug(f"  Result {i+1}: distance={distance:.4f}, similarity={similarity:.4f}, type={record.chunk_type}")

                if similarity >= threshold:
                    formatted.append({
                        "id": record.id,
                        "paper_id": record.paper_id,
                        "chunk_type": record.chunk_type,
                        "content": record.content[:200] if record.content else "",
                        "similarity": float(similarity),
                        "metadata": record.chunk_metadata
                    })

            # Limit to requested amount after threshold filtering
            formatted = formatted[:limit]

            logger.debug(f"After threshold {threshold:.2f}: {len(formatted)} results")

            return formatted

        except Exception as e:
            logger.error(f"Search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            session.close()

    def get_stats(self) -> Dict:
        """Get database statistics"""
        session = self.Session()

        try:
            total = session.query(sa.func.count(PaperEmbedding.id)).scalar()
            by_type = session.query(
                PaperEmbedding.chunk_type,
                sa.func.count(PaperEmbedding.id)
            ).group_by(PaperEmbedding.chunk_type).all()

            papers = session.query(sa.func.count(sa.distinct(PaperEmbedding.paper_id))).scalar()

            return {
                "total_embeddings": total,
                "unique_papers": papers,
                "by_type": dict(by_type)
            }

        finally:
            session.close()

    def delete_paper(self, paper_id: str) -> int:
        """Delete all embeddings for a paper"""
        session = self.Session()

        try:
            deleted = session.query(PaperEmbedding).filter(
                PaperEmbedding.paper_id == paper_id
            ).delete()
            session.commit()
            logger.info(f"Deleted {deleted} embeddings for paper {paper_id}")
            return deleted

        finally:
            session.close()

    def export_to_json(self, output_path: str, paper_id: str = None) -> str:
        """Export embeddings to JSON"""
        session = self.Session()

        try:
            query = session.query(PaperEmbedding)
            if paper_id:
                query = query.filter(PaperEmbedding.paper_id == paper_id)

            records = []
            for record in query.all():
                records.append({
                    "id": record.id,
                    "paper_id": record.paper_id,
                    "chunk_type": record.chunk_type,
                    "content": record.content,
                    "metadata": record.chunk_metadata,
                    "created_at": record.created_at.isoformat()
                })

            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2)

            logger.info(f"✓ Exported {len(records)} records to {output_path}")
            return output_path

        finally:
            session.close()


def setup_local_postgres():
    """
    Setup instructions for local PostgreSQL with pgvector
    """
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  PostgreSQL + pgvector Setup Instructions                  ║
    ╚════════════════════════════════════════════════════════════╝

    1. Install PostgreSQL (macOS):
       brew install postgresql

    2. Install pgvector extension:
       brew install pgvector

    3. Start PostgreSQL:
       brew services start postgresql

    4. Create database:
       createdb embed_rag

    5. Enable pgvector extension:
       psql embed_rag -c "CREATE EXTENSION IF NOT EXISTS vector"

    6. Create user (optional):
       psql embed_rag -c "CREATE USER embed_user WITH PASSWORD 'password'"
       psql embed_rag -c "GRANT ALL ON DATABASE embed_rag TO embed_user"

    7. Connection string:
       postgresql://postgres@localhost:5432/embed_rag
       OR
       postgresql://embed_user:password@localhost:5432/embed_rag

    Environment variable:
       export DATABASE_URL="postgresql://postgres@localhost:5432/embed_rag"
    """)


def main():
    """Test vector store"""

    print("\n" + "="*60)
    print("VECTOR STORE TEST")
    print("="*60)

    # Try to connect
    try:
        store = VectorStore()

        # Get stats
        stats = store.get_stats()
        print(f"\nDatabase Stats:")
        print(f"  Total embeddings: {stats['total_embeddings']}")
        print(f"  Unique papers: {stats['unique_papers']}")
        print(f"  By type: {stats['by_type']}")

    except Exception as e:
        logger.warning(f"Database not available: {e}")
        print("\nDatabase not configured. Run setup:")
        setup_local_postgres()

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
