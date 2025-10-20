#!/usr/bin/env python3
"""
Ensemble Embedding Module
Combines SPECTER2 + E5-Large + SciBERT for optimal scientific paper embeddings
"""

import os
import sys
from typing import List, Dict, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>")


class EnsembleEmbeddingModel:
    """
    Ensemble of three models:
    1. SPECTER2 (768-dim): Scientific papers specialist
    2. E5-Large (1024-dim): Multilingual + high quality
    3. SciBERT (768-dim): Scientific vocabulary expert
    """

    def __init__(self, device: str = None, cache_dir: str = None):
        """Initialize ensemble models"""
        if device is None:
            # Apple Silicon (M1/M2/M3/M4) için MPS kullan
            if torch.backends.mps.is_available():
                device = "mps"  # Metal Performance Shaders - GPU acceleration
            elif torch.cuda.is_available():
                device = "cuda"  # NVIDIA GPU
            else:
                device = "cpu"  # Fallback

        self.device = device
        self.cache_dir = cache_dir or "./model_cache"

        logger.info(f"Device: {self.device}")
        logger.info(f"Cache directory: {self.cache_dir}")

        # Create cache dir
        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info("Loading models...")

        try:
            # Model 1: SPECTER2 (best for scientific papers)
            logger.info("  Loading SPECTER2...")
            self.specter2 = SentenceTransformer(
                "allenai/specter2_base",
                device=self.device,
                cache_folder=self.cache_dir
            )
            self.specter2_dim = 768

            # Model 2: E5-Large (multilingual)
            logger.info("  Loading E5-Large (multilingual)...")
            self.e5_large = SentenceTransformer(
                "intfloat/multilingual-e5-large",
                device=self.device,
                cache_folder=self.cache_dir
            )
            self.e5_large_dim = 1024

            # Model 3: SciBERT
            logger.info("  Loading SciBERT...")
            self.scibert = SentenceTransformer(
                "allenai/scibert_scivocab_uncased",
                device=self.device,
                cache_folder=self.cache_dir
            )
            self.scibert_dim = 768

            logger.info("✓ All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def embed_specter2(self, texts: List[str]) -> np.ndarray:
        """Generate SPECTER2 embeddings (768-dim)"""
        embeddings = self.specter2.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

    def embed_e5_large(self, texts: List[str]) -> np.ndarray:
        """Generate E5-Large embeddings (1024-dim)"""
        # E5 models work better with task instructions
        query_texts = [f"query: {text}" for text in texts]

        embeddings = self.e5_large.encode(
            query_texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

    def embed_scibert(self, texts: List[str]) -> np.ndarray:
        """Generate SciBERT embeddings (768-dim)"""
        embeddings = self.scibert.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalization"""
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def embed_ensemble(
        self,
        texts: List[str],
        weights: Tuple[float, float, float] = (0.2, 0.6, 0.2),
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate ensemble embeddings using weighted combination

        Args:
            texts: List of texts to embed
            weights: (specter2_weight, e5_large_weight, scibert_weight)
                    Default weights optimized via grid search (2025-10-19):
                    - E5-Large: 0.6 (multilingual, general-purpose strength)
                    - SPECTER2: 0.2 (scientific domain)
                    - SciBERT: 0.2 (scientific vocabulary)
                    Semantic similarity improvement: 60% → 66.7% (+11.2%)
            normalize: Whether to normalize final embeddings

        Returns:
            Combined embeddings (1024-dim, matching E5-Large dimensions)
        """
        assert sum(weights) == 1.0, "Weights must sum to 1.0"

        logger.info(f"Embedding {len(texts)} texts with ensemble...")

        # Get individual embeddings
        specter2_emb = self.embed_specter2(texts)  # (N, 768)
        e5_large_emb = self.embed_e5_large(texts)  # (N, 1024)
        scibert_emb = self.embed_scibert(texts)  # (N, 768)

        logger.info(f"  SPECTER2: {specter2_emb.shape}")
        logger.info(f"  E5-Large: {e5_large_emb.shape}")
        logger.info(f"  SciBERT: {scibert_emb.shape}")

        # Normalize each embedding
        specter2_norm = self._normalize(specter2_emb)
        e5_large_norm = self._normalize(e5_large_emb)
        scibert_norm = self._normalize(scibert_emb)

        # Simple weighted average approach:
        # Project 768-dim models to 1024-dim by padding with zeros, then average

        # Pad SPECTER2 and SciBERT to 1024 dims
        specter2_padded = np.hstack([
            specter2_norm,
            np.zeros((specter2_norm.shape[0], 256))  # Pad to 1024
        ])

        scibert_padded = np.hstack([
            scibert_norm,
            np.zeros((scibert_norm.shape[0], 256))  # Pad to 1024
        ])

        # Weighted average: apply weights to each 1024-dim embedding
        final_embeddings = (
            specter2_padded * weights[0] +
            e5_large_norm * weights[1] +
            scibert_padded * weights[2]
        )

        if normalize:
            final_embeddings = self._normalize(final_embeddings)

        logger.info(f"✓ Final embedding shape: {final_embeddings.shape}")

        return final_embeddings

    def embed_by_type(
        self,
        text: str,
        chunk_type: str = "text"
    ) -> np.ndarray:
        """
        Type-aware embedding with specialized handling

        Args:
            text: Text to embed
            chunk_type: "text", "formula", "table", or "theorem"

        Returns:
            Embedding (1024-dim)
        """
        # Add type prefix for better context
        type_prefixes = {
            "text": "scientific text: ",
            "formula": "mathematical formula: ",
            "table": "data table: ",
            "theorem": "theorem statement: "
        }

        prefixed_text = type_prefixes.get(chunk_type, "") + text

        return self.embed_ensemble([prefixed_text], normalize=True)[0]

    def batch_embed(
        self,
        chunks: List[Dict],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Embed multiple chunks with metadata preservation

        Args:
            chunks: List of {"id": str, "content": str, "type": str}
            batch_size: Batch size for processing

        Returns:
            List of {"id": str, "content": str, "embedding": array, ...}
        """
        logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}...")

        texts = [chunk.get("content", chunk.get("latex", "")) for chunk in chunks]
        types = [chunk.get("type", "text") for chunk in chunks]

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_types = types[i:i+batch_size]

            # Add type prefixes
            prefixed = [
                f"{{'type': '{t}'}} " + text[:100]
                for t, text in zip(batch_types, batch_texts)
            ]

            batch_emb = self.embed_ensemble(prefixed, normalize=True)
            all_embeddings.append(batch_emb)

        embeddings = np.vstack(all_embeddings)

        # Combine with metadata
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            result.append({
                **chunk,
                "embedding": embedding.tolist(),
                "embedding_model": "ensemble_specter2_e5_scibert",
                "embedding_dim": len(embedding)
            })

        logger.info(f"✓ Processed {len(result)} chunks")
        return result


def main():
    """Test ensemble embedding model"""

    # Sample texts from scientific paper
    sample_texts = [
        "The rough set theory provides an effective tool for dealing with uncertainty.",
        "\\min \\sum_{i,j,k} c_{ijk} x_{ijk}",
        "Table 1 shows the transportation costs for different routes and vehicles.",
        "Theorem 1: Let (U, R) be an approximation space. Then the following properties hold."
    ]

    # Initialize ensemble
    ensemble = EnsembleEmbeddingModel()

    # Generate embeddings
    print("\n" + "="*60)
    print("ENSEMBLE EMBEDDING TEST")
    print("="*60)

    embeddings = ensemble.embed_ensemble(sample_texts, normalize=True)

    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    print(f"Each embedding dimension: {embeddings.shape[1]}\n")

    for i, (text, emb) in enumerate(zip(sample_texts, embeddings)):
        print(f"Sample {i+1}:")
        print(f"  Text: {text[:50]}...")
        print(f"  Embedding norm: {np.linalg.norm(emb):.4f}")
        print(f"  First 5 dims: {emb[:5]}")
        print()

    # Test type-aware embedding
    print("="*60)
    print("TYPE-AWARE EMBEDDING TEST")
    print("="*60)

    text = "Minimize total transportation cost"
    for chunk_type in ["text", "formula", "table", "theorem"]:
        emb = ensemble.embed_by_type(text, chunk_type)
        print(f"{chunk_type:8} | Dim: {len(emb)}, Norm: {np.linalg.norm(emb):.4f}")

    print("="*60)


if __name__ == "__main__":
    main()
