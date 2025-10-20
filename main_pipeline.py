#!/usr/bin/env python3
"""
Main RAG Pipeline - Complete end-to-end processing
PDF → DeepDoc → Chunks → Embeddings → Vector Store
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from loguru import logger
import numpy as np

from deepdoc_test import DeepDocProcessor
from ensemble_embeddings import EnsembleEmbeddingModel
from vector_store import VectorStore
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import re

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG"
)
logger.add(
    "pipeline.log",
    format="<level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="DEBUG"
)


class RAGPipeline:
    """Complete RAG pipeline orchestrator"""

    def __init__(
        self,
        db_url: str = None,
        device: str = None,
        cache_dir: str = "./model_cache",
        create_index: bool = False,
        index_lists: int = None
    ):
        """
        Initialize pipeline components

        Args:
            db_url: PostgreSQL connection string
            device: Device for embedding models (cuda/cpu)
            cache_dir: Model cache directory
            create_index: Whether to create IVFFlat index (default False)
                         For large datasets (40GB+), create index AFTER loading all data
                         Use pipeline.create_index(lists=2000) after processing
            index_lists: Number of lists for IVFFlat index (auto-calculated if None)
        """

        logger.info("Initializing RAG Pipeline...")

        # Initialize components
        self.deepdoc = None
        self.embedding_model = None
        self.vector_store = None
        self.reranker = None  # Lazy-loaded cross-encoder for reranking
        self.bm25_index = None  # Lazy-loaded BM25 index for keyword search
        self.bm25_corpus = []  # Document corpus for BM25
        self.bm25_metadata = []  # Metadata mapping for BM25 results

        try:
            logger.info("Loading embedding models...")
            self.embedding_model = EnsembleEmbeddingModel(
                device=device,
                cache_dir=cache_dir
            )

            logger.info("Connecting to vector store...")
            self.vector_store = VectorStore(
                db_url=db_url,
                create_tables=True,
                create_index=create_index,
                index_lists=index_lists
            )

            logger.info("✓ Pipeline initialized successfully\n")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise

    def process_pdf(self, pdf_path: str, paper_id: str = None) -> Dict:
        """
        Process PDF and store embeddings

        Args:
            pdf_path: Path to PDF file
            paper_id: Unique paper identifier (auto-generated if None)

        Returns:
            Processing results
        """

        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return {"error": "PDF not found"}

        if paper_id is None:
            paper_id = Path(pdf_path).stem

        logger.info(f"Processing: {Path(pdf_path).name}")
        logger.info(f"Paper ID: {paper_id}\n")

        # Step 1: Extract components with DeepDoc
        logger.info("┌─ STEP 1: Extract chunks (DeepDoc)")
        processor = DeepDocProcessor(pdf_path, paper_id=paper_id)
        chunks_data = processor.extract_from_pdf()

        text_chunks = chunks_data.get("text", [])
        formula_chunks = chunks_data.get("formula", [])
        table_chunks = chunks_data.get("table", [])

        logger.info(f"├─ Text chunks: {len(text_chunks)}")
        logger.info(f"├─ Formula chunks: {len(formula_chunks)}")
        logger.info(f"└─ Table chunks: {len(table_chunks)}")
        logger.info(f"└─ ✓ Extraction complete\n")

        # Step 2: Generate embeddings
        logger.info("┌─ STEP 2: Generate embeddings (Ensemble)")

        all_records = []

        # Process text chunks
        if text_chunks:
            logger.info(f"├─ Embedding {len(text_chunks)} text chunks...")
            text_texts = [
                chunk.get("content", "") for chunk in text_chunks
            ]
            text_embeddings = self.embedding_model.embed_ensemble(
                text_texts,
                normalize=True
            )

            for i, (chunk, embedding) in enumerate(zip(text_chunks, text_embeddings)):
                record = {
                    "id": chunk["id"],
                    "paper_id": paper_id,
                    "chunk_type": "text",
                    "content": chunk.get("content", ""),
                    "embedding": embedding.tolist(),
                    "metadata": {
                        "page": chunk.get("page", 0),
                        "section": chunk.get("section", ""),
                        "source": "text_extraction"
                    }
                }
                all_records.append(record)

        # Process formula chunks
        if formula_chunks:
            logger.info(f"├─ Embedding {len(formula_chunks)} formula chunks...")
            formula_texts = []
            for chunk in formula_chunks:
                # Combine LaTeX with description for better embedding
                text = f"Formula: {chunk.get('latex', '')} | {chunk.get('description', '')}"
                formula_texts.append(text)

            formula_embeddings = self.embedding_model.embed_ensemble(
                formula_texts,
                normalize=True
            )

            for i, (chunk, embedding) in enumerate(zip(formula_chunks, formula_embeddings)):
                record = {
                    "id": chunk["id"],
                    "paper_id": paper_id,
                    "chunk_type": "formula",
                    "content": chunk.get("description", ""),
                    "latex_content": chunk.get("latex", ""),
                    "embedding": embedding.tolist(),
                    "metadata": {
                        "page": chunk.get("page", 0),
                        "latex": chunk.get("latex", ""),
                        "context_before": chunk.get("context_before", "")[:100],
                        "source": "formula_extraction"
                    }
                }
                all_records.append(record)

        # Process table chunks
        if table_chunks:
            logger.info(f"├─ Embedding {len(table_chunks)} table chunks...")
            table_texts = []
            for chunk in table_chunks:
                # Use description + raw content for embedding
                text = f"Table: {chunk.get('description', '')} | {chunk.get('raw_content', '')[:200]}"
                table_texts.append(text)

            table_embeddings = self.embedding_model.embed_ensemble(
                table_texts,
                normalize=True
            )

            for i, (chunk, embedding) in enumerate(zip(table_chunks, table_embeddings)):
                record = {
                    "id": chunk["id"],
                    "paper_id": paper_id,
                    "chunk_type": "table",
                    "content": chunk.get("description", ""),
                    "embedding": embedding.tolist(),
                    "metadata": {
                        "page": chunk.get("page", 0),
                        "rows": chunk.get("rows", 0),
                        "cols": chunk.get("cols", 0),
                        "source": "table_extraction"
                    }
                }
                all_records.append(record)

        logger.info(f"└─ ✓ Generated {len(all_records)} embeddings\n")

        # Step 3: Store in vector database
        logger.info("┌─ STEP 3: Store embeddings (PostgreSQL + pgvector)")
        inserted = self.vector_store.insert_batch(all_records)
        logger.info(f"├─ Inserted: {inserted} records")

        # Get stats
        stats = self.vector_store.get_stats()
        logger.info(f"├─ Database total: {stats['total_embeddings']} embeddings")
        logger.info(f"└─ ✓ Storage complete\n")

        # Summary
        result = {
            "paper_id": paper_id,
            "pdf_path": pdf_path,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "extraction": {
                "text_chunks": len(text_chunks),
                "formula_chunks": len(formula_chunks),
                "table_chunks": len(table_chunks),
                "total_chunks": len(all_records)
            },
            "storage": {
                "inserted": inserted,
                "total_embeddings": stats['total_embeddings'],
                "by_type": stats['by_type']
            }
        }

        logger.info("✓ Pipeline complete!\n")
        return result

    def search(
        self,
        query: str,
        limit: int = 5,
        chunk_type: str = None,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search embeddings using semantic similarity

        Args:
            query: Natural language query
            limit: Number of results
            chunk_type: Filter by type (optional)
            threshold: Minimum similarity score (0.0-1.0)
                      Default 0.0 to return all results with scores
                      Typical good matches: 0.71-0.83
                      Raise threshold to filter more aggressively

        Returns:
            List of similar chunks with similarity scores
        """

        logger.info(f"Searching: '{query}'")
        logger.debug(f"  Limit: {limit}, Type: {chunk_type}, Threshold: {threshold}")

        # Generate query embedding
        query_embedding = self.embedding_model.embed_ensemble(
            [query],
            normalize=True
        )[0]

        logger.debug(f"  Query embedding shape: {query_embedding.shape}")
        logger.debug(f"  Query embedding norm: {np.linalg.norm(query_embedding):.4f}")

        # Search
        results = self.vector_store.search_similar(
            query_embedding.tolist(),
            limit=limit,
            chunk_type=chunk_type,
            threshold=threshold
        )

        logger.info(f"Found {len(results)} results")
        if results:
            logger.debug(f"  Top similarity: {results[0]['similarity']:.4f}")
            logger.debug(f"  Lowest similarity: {results[-1]['similarity']:.4f}")
        logger.info("")

        return results

    def _load_reranker(self, device: str = None):
        """
        Lazy-load cross-encoder for reranking

        Args:
            device: Device to load model on (mps/cuda/cpu)
        """
        if self.reranker is None:
            logger.info("Loading cross-encoder for reranking...")

            if device is None:
                # Use same device as embedding model
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

            self.reranker = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2',
                device=device
            )
            logger.info(f"✓ Cross-encoder loaded on {device}")

    def search_with_rerank(
        self,
        query: str,
        limit: int = 5,
        rerank_top_k: int = 20,
        chunk_type: str = None,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Two-stage retrieval: Fast embedding search + Precise cross-encoder reranking

        Stage 1 (Fast): Use embedding similarity to get top-K candidates
        Stage 2 (Precise): Use cross-encoder to rerank and return top-N

        Args:
            query: Natural language query
            limit: Final number of results to return
            rerank_top_k: Number of candidates to retrieve before reranking (default 20)
            chunk_type: Filter by type (optional)
            threshold: Minimum similarity score for initial retrieval

        Returns:
            Reranked list of chunks with cross-encoder scores

        Example:
            # Get top-5 results with high precision
            results = pipeline.search_with_rerank(
                "What is rough set theory?",
                limit=5,
                rerank_top_k=20  # Retrieve 20, rerank, return top-5
            )
        """

        logger.info(f"Searching with reranking: '{query}'")
        logger.debug(f"  Initial retrieval: top-{rerank_top_k}")
        logger.debug(f"  Final results: top-{limit}")

        # Stage 1: Fast retrieval with embeddings
        candidates = self.search(
            query=query,
            limit=rerank_top_k,
            chunk_type=chunk_type,
            threshold=threshold
        )

        if not candidates:
            logger.warning("No candidates found for reranking")
            return []

        logger.info(f"Stage 1: Retrieved {len(candidates)} candidates")
        logger.debug(f"  Embedding similarity range: {candidates[0]['similarity']:.3f} - {candidates[-1]['similarity']:.3f}")

        # Stage 2: Rerank with cross-encoder
        self._load_reranker()

        logger.info("Stage 2: Reranking with cross-encoder...")

        # Prepare query-passage pairs
        pairs = [(query, c['content']) for c in candidates]

        # Get cross-encoder scores
        rerank_scores = self.reranker.predict(pairs)

        # Attach scores and sort
        for candidate, score in zip(candidates, rerank_scores):
            candidate['rerank_score'] = float(score)
            candidate['original_rank'] = candidates.index(candidate) + 1

        # Sort by rerank score (descending)
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        # Take top-N
        final_results = reranked[:limit]

        logger.info(f"✓ Reranking complete")
        logger.debug(f"  Top rerank score: {final_results[0]['rerank_score']:.3f}")
        logger.debug(f"  Top result original rank: #{final_results[0]['original_rank']}")
        logger.info("")

        return final_results

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer for BM25

        Args:
            text: Text to tokenize

        Returns:
            List of lowercased tokens
        """
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _load_bm25_index(self):
        """
        Build BM25 index from database corpus

        Loads all documents from database and creates BM25 index for keyword search
        """
        if self.bm25_index is not None:
            return  # Already loaded

        logger.info("Building BM25 index from corpus...")

        # Get all embeddings from database
        from vector_store import PaperEmbedding
        session = self.vector_store.Session()

        try:
            all_records = session.query(PaperEmbedding).all()

            if not all_records:
                logger.warning("No documents found for BM25 index")
                return

            # Build corpus
            self.bm25_corpus = []
            self.bm25_metadata = []

            for record in all_records:
                content = record.content or ""
                # Tokenize
                tokens = self._tokenize(content)
                self.bm25_corpus.append(tokens)

                # Store metadata
                self.bm25_metadata.append({
                    "id": record.id,
                    "paper_id": record.paper_id,
                    "chunk_type": record.chunk_type,
                    "content": content,
                    "metadata": record.chunk_metadata
                })

            # Build BM25 index
            self.bm25_index = BM25Okapi(self.bm25_corpus)

            logger.info(f"✓ BM25 index built with {len(self.bm25_corpus)} documents")

        finally:
            session.close()

    def search_hybrid(
        self,
        query: str,
        limit: int = 5,
        chunk_type: str = None,
        threshold: float = 0.0,
        bm25_weight: float = 0.1,
        embedding_weight: float = 0.9,
        use_reranking: bool = True
    ) -> List[Dict]:
        """
        Hybrid search: BM25 keyword search + Embedding semantic search

        Combines keyword matching (BM25) with semantic similarity (embeddings)
        using Reciprocal Rank Fusion (RRF) for merging results

        Args:
            query: Natural language query
            limit: Final number of results to return
            chunk_type: Filter by type (optional)
            threshold: Minimum similarity score for embedding search
            bm25_weight: Weight for BM25 scores (default 0.1)
                        Optimized via grid search (2025-10-19):
                        - Tested 12 combinations from pure BM25 (1.0/0.0) to pure embedding (0.0/1.0)
                        - Result: Embedding-dominant configuration performs best
                        - Semantic similarity improvement: 46.7% → 66.7% (+42.8%)
                        - For scientific papers, semantic understanding >> keyword matching
            embedding_weight: Weight for embedding scores (default 0.9)
                             High weight prioritizes semantic similarity over exact keyword matches
                             Optimal for queries requiring concept understanding vs literal matching
            use_reranking: Whether to use cross-encoder reranking (default True)

        Returns:
            Hybrid search results with combined scores

        Performance Notes:
            - Pure embedding (0.0/1.0): 66.7% semantic similarity
            - Optimal hybrid (0.1/0.9): 66.7% semantic similarity
            - Baseline (0.3/0.7): 46.7% semantic similarity ❌
            - Equal balance (0.5/0.5): 13.3% semantic similarity ❌
            - BM25-heavy (0.6/0.4): 0% semantic similarity ❌

        Example:
            # Default: Semantic-focused hybrid search
            results = pipeline.search_hybrid(
                "rough set theory applications",
                limit=5
            )

            # Keyword-heavy query (rare use case)
            results = pipeline.search_hybrid(
                "exact term matching needed",
                limit=5,
                bm25_weight=0.3,  # Increase for exact keyword importance
                embedding_weight=0.7
            )
        """

        logger.info(f"Hybrid search: '{query}'")
        logger.debug(f"  BM25 weight: {bm25_weight}, Embedding weight: {embedding_weight}")

        # Load BM25 index
        self._load_bm25_index()

        if not self.bm25_index:
            logger.warning("BM25 index not available, falling back to embedding-only search")
            if use_reranking:
                return self.search_with_rerank(query, limit, chunk_type=chunk_type, threshold=threshold)
            else:
                return self.search(query, limit, chunk_type=chunk_type, threshold=threshold)

        # 1. BM25 keyword search
        logger.info("Stage 1: BM25 keyword search...")
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # Get top-K BM25 results
        top_k_bm25 = min(50, len(bm25_scores))
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]

        bm25_results = []
        for idx in bm25_indices:
            if bm25_scores[idx] > 0:  # Only include non-zero scores
                result = self.bm25_metadata[idx].copy()
                result['bm25_score'] = float(bm25_scores[idx])
                result['bm25_rank'] = len(bm25_results) + 1
                bm25_results.append(result)

        logger.info(f"  BM25 found {len(bm25_results)} results")

        # 2. Embedding semantic search
        logger.info("Stage 2: Embedding semantic search...")
        embedding_results = self.search(
            query=query,
            limit=50,
            chunk_type=chunk_type,
            threshold=threshold
        )

        for i, result in enumerate(embedding_results):
            result['embedding_rank'] = i + 1

        logger.info(f"  Embedding found {len(embedding_results)} results")

        # 3. Merge with Reciprocal Rank Fusion (RRF)
        logger.info("Stage 3: Merging results with RRF...")

        # Combine results by ID
        combined = {}
        k = 60  # RRF constant

        # Add BM25 results
        for result in bm25_results:
            doc_id = result['id']
            rrf_score = 1.0 / (k + result['bm25_rank'])
            combined[doc_id] = {
                **result,
                'rrf_bm25': rrf_score * bm25_weight,
                'rrf_embedding': 0.0
            }

        # Add embedding results
        for result in embedding_results:
            doc_id = result['id']
            rrf_score = 1.0 / (k + result['embedding_rank'])

            if doc_id in combined:
                combined[doc_id]['rrf_embedding'] = rrf_score * embedding_weight
                combined[doc_id]['similarity'] = result['similarity']
            else:
                combined[doc_id] = {
                    **result,
                    'rrf_bm25': 0.0,
                    'rrf_embedding': rrf_score * embedding_weight,
                    'bm25_score': 0.0
                }

        # Calculate final hybrid score
        for doc_id in combined:
            combined[doc_id]['hybrid_score'] = (
                combined[doc_id]['rrf_bm25'] +
                combined[doc_id]['rrf_embedding']
            )

        # Sort by hybrid score
        merged_results = sorted(
            combined.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )

        logger.info(f"  Merged to {len(merged_results)} unique results")

        # 4. Optional reranking
        if use_reranking and len(merged_results) > 0:
            logger.info("Stage 4: Cross-encoder reranking...")

            self._load_reranker()

            # Take top candidates for reranking
            top_k_rerank = min(20, len(merged_results))
            candidates = merged_results[:top_k_rerank]

            # Prepare pairs
            pairs = [(query, c['content']) for c in candidates]
            rerank_scores = self.reranker.predict(pairs)

            # Attach rerank scores
            for candidate, score in zip(candidates, rerank_scores):
                candidate['rerank_score'] = float(score)

            # Sort by rerank score
            candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

            final_results = candidates[:limit]

            logger.info(f"✓ Hybrid search complete (with reranking)")
        else:
            final_results = merged_results[:limit]
            logger.info(f"✓ Hybrid search complete")

        logger.info("")

        return final_results

    def search_multi_query(
        self,
        query: str,
        limit: int = 5,
        chunk_type: str = None,
        threshold: float = 0.0,
        bm25_weight: float = 0.1,
        embedding_weight: float = 0.9,
        use_reranking: bool = True,
        max_variants: int = 5
    ) -> List[Dict]:
        """
        Multi-query search with query expansion

        Generates multiple query variants using domain keywords, then merges results
        using Reciprocal Rank Fusion (RRF) for improved recall.

        Args:
            query: Original natural language query
            limit: Final number of results to return
            chunk_type: Filter by type (optional)
            threshold: Minimum similarity score for embedding search
            bm25_weight: Weight for BM25 scores (default 0.1)
            embedding_weight: Weight for embedding scores (default 0.9)
            use_reranking: Whether to use cross-encoder reranking (default True)
            max_variants: Maximum query variants to generate (default 5)

        Returns:
            Merged search results with RRF scores

        Performance Benefits:
            - Query expansion improves recall for semantic variations
            - Multiple query formulations capture different aspects
            - RRF merging combines evidence from all variants
            - Expected improvement: +8-13% semantic similarity

        Example:
            # Single query becomes multiple variants:
            "minimize cost" →
                1. "minimize cost" (original)
                2. "minimization cost" (synonym)
                3. "cost minimization" (reordering)
                4. "minimize cost minimize minimization" (additive)

            results = pipeline.search_multi_query(
                "minimize transportation cost",
                limit=5,
                max_variants=5
            )
        """
        # Import query expander
        from query_expansion import QueryExpander

        logger.info(f"Multi-query search: '{query}'")

        # Initialize query expander (lazy loading)
        if not hasattr(self, '_query_expander'):
            try:
                self._query_expander = QueryExpander()
            except Exception as e:
                logger.warning(f"Query expander failed to load: {e}")
                logger.warning("Falling back to single query search")
                return self.search_hybrid(
                    query=query,
                    limit=limit,
                    chunk_type=chunk_type,
                    threshold=threshold,
                    bm25_weight=bm25_weight,
                    embedding_weight=embedding_weight,
                    use_reranking=use_reranking
                )

        # Generate query variants
        logger.info("Generating query variants...")
        variants = self._query_expander.generate_query_variants(
            query,
            strategies=["original", "synonym", "additive"]
        )

        # Limit number of variants
        variants = variants[:max_variants]

        logger.info(f"  Generated {len(variants)} query variants:")
        for i, v in enumerate(variants, 1):
            logger.info(f"    {i}. [{v['strategy']:8}] {v['query']}")

        # Search with each variant
        all_results = {}  # {doc_id: {result_data, ranks: [rank1, rank2, ...]}}

        for i, variant in enumerate(variants):
            variant_query = variant["query"]
            logger.info(f"\n  Searching variant {i+1}/{len(variants)}: '{variant_query[:50]}...'")

            # Get results for this variant
            results = self.search_hybrid(
                query=variant_query,
                limit=limit * 3,  # Get more results for better merging
                chunk_type=chunk_type,
                threshold=threshold,
                bm25_weight=bm25_weight,
                embedding_weight=embedding_weight,
                use_reranking=use_reranking
            )

            # Store results with ranks
            for rank, result in enumerate(results, 1):
                doc_id = result.get("id")
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "data": result,
                        "ranks": []
                    }
                all_results[doc_id]["ranks"].append(rank)

        # Merge results using Reciprocal Rank Fusion (RRF)
        logger.info(f"\n  Merging {len(all_results)} unique results with RRF...")

        k = 60  # RRF constant (standard value)
        for doc_id in all_results:
            ranks = all_results[doc_id]["ranks"]
            # RRF score: sum of 1/(k + rank) for all ranks
            rrf_score = sum(1.0 / (k + r) for r in ranks)
            all_results[doc_id]["rrf_score"] = rrf_score

        # Sort by RRF score (descending)
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # Extract top results
        final_results = []
        for item in sorted_results[:limit]:
            result = item["data"].copy()
            result["rrf_score"] = item["rrf_score"]
            result["num_variants_matched"] = len(item["ranks"])
            final_results.append(result)

        logger.info(f"✓ Multi-query search complete")
        logger.info(f"  Returned {len(final_results)} results (merged from {len(all_results)} unique documents)")
        logger.info("")

        return final_results

    def create_index(self, lists: int = None, force: bool = False):
        """
        Create IVFFlat index after loading all data

        Args:
            lists: Number of lists for IVFFlat (default: auto-calculated)
                  Recommended for large datasets:
                  - 100K-1M vectors: 500-1000 lists
                  - 1M-10M vectors: 1000-3000 lists
                  - 10M+ vectors: 3000-5000 lists
            force: Create even if <1000 vectors

        Example:
            # After processing 40GB of documents
            pipeline = RAGPipeline(create_index=False)
            # ... process many PDFs ...
            pipeline.create_index(lists=2000)  # Create index at the end
        """
        logger.info("Creating vector index...")
        self.vector_store.create_index_now(lists=lists, force=force)
        logger.info("✓ Index creation complete")

    def print_results(self, results: List[Dict]):
        """Pretty print search results"""
        print("\n" + "="*70)
        print("SEARCH RESULTS")
        print("="*70 + "\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['chunk_type'].upper()}] (Similarity: {result['similarity']:.2%})")
            print(f"   Content: {result['content'][:100]}...")
            print(f"   Metadata: {result['metadata']}")
            print()

        print("="*70 + "\n")


def main():
    """Main execution"""

    print("\n" + "="*70)
    print("RAG PIPELINE - Scientific Paper Embedding & Retrieval")
    print("="*70 + "\n")

    # Initialize pipeline
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.info("Make sure:")
        logger.info("  1. PostgreSQL is running with pgvector extension")
        logger.info("  2. DATABASE_URL environment variable is set")
        logger.info("  3. Internet connection available for downloading models")
        return

    # Process sample paper
    pdf_path = "/Users/bekiragirgun/Downloads/Documents/aktarılan/A class of rough multiple objective programming and its application to solid transportation problem.pdf"

    if os.path.exists(pdf_path):
        result = pipeline.process_pdf(pdf_path)

        print("\n" + "="*70)
        print("PROCESSING RESULT")
        print("="*70)
        print(json.dumps(result, indent=2))
        print("="*70 + "\n")

        # Test search
        if result.get("status") == "success":
            print("Testing semantic search...\n")

            queries = [
                "What is rough set theory?",
                "Minimize transportation cost",
                "Transportation constraints",
            ]

            for query in queries:
                print(f"Query: {query}")
                results = pipeline.search(query, limit=3)
                pipeline.print_results(results)

    else:
        logger.error(f"Sample PDF not found: {pdf_path}")


if __name__ == "__main__":
    main()
