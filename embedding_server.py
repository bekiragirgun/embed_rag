#!/usr/bin/env python3
"""
OpenAI-compatible Embedding Server
SPECTER2 + E5-Large + SciBERT Ensemble
"""

import os
import sys
import json
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger

from ensemble_embeddings import EnsembleEmbeddingModel

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<level>{level: <8}</level> | <level>{message}</level>"
)

# Initialize FastAPI app
app = FastAPI(
    title="EmbedRAG Embedding Server",
    description="OpenAI-compatible embedding API with SPECTER2 + E5-Large + SciBERT",
    version="1.0.0"
)

# Global embedding model
embedding_model: Optional[EnsembleEmbeddingModel] = None


# ============================================================================
# Pydantic Models (OpenAI-compatible format)
# ============================================================================

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: List[str] | str
    model: str = "ensemble"
    encoding_format: str = "float"


class Embedding(BaseModel):
    """Individual embedding in OpenAI format"""
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[Embedding]
    model: str
    usage: dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    version: str
    timestamp: str


# ============================================================================
# Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize embedding model on startup"""
    global embedding_model

    logger.info("Starting EmbedRAG Embedding Server...")

    try:
        device = os.getenv("EMBEDDING_MODEL_DEVICE", "cpu")
        logger.info(f"Loading ensemble models (device: {device})...")

        embedding_model = EnsembleEmbeddingModel(device=device)

        logger.info("✓ Embedding models loaded successfully")
        logger.info(f"Available models: SPECTER2, E5-Large, SciBERT")
        logger.info(f"Embedding dimension: 1024")

    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    return HealthResponse(
        status="healthy",
        model="ensemble_specter2_e5_scibert",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings using OpenAI-compatible format

    Example:
        curl -X POST "http://localhost:8000/v1/embeddings" \
            -H "Content-Type: application/json" \
            -d '{
                "input": "Hello world",
                "model": "ensemble"
            }'
    """

    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Normalize input
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    if not texts:
        raise HTTPException(status_code=400, detail="Input texts are empty")

    try:
        logger.info(f"Embedding {len(texts)} text(s)...")

        # Generate embeddings
        embeddings = embedding_model.embed_ensemble(
            texts,
            normalize=True
        )

        # Format response in OpenAI style
        data = []
        for i, embedding in enumerate(embeddings):
            data.append(Embedding(
                index=i,
                embedding=embedding.tolist()
            ))

        # Calculate tokens (rough estimate)
        total_tokens = sum(len(text.split()) for text in texts) * 1.3

        response = EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": int(total_tokens),
                "total_tokens": int(total_tokens)
            }
        )

        logger.info(f"✓ Generated {len(embeddings)} embedding(s)")
        return response

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings/batch")
async def batch_embeddings(request: dict):
    """
    Batch embedding endpoint for multiple texts

    Expected format:
    {
        "inputs": [
            {"text": "...", "type": "text"},
            {"text": "...", "type": "formula"}
        ]
    }
    """

    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        inputs = request.get("inputs", [])

        if not inputs:
            raise ValueError("No inputs provided")

        logger.info(f"Processing batch of {len(inputs)} items...")

        results = []
        for i, item in enumerate(inputs):
            text = item.get("text", "")
            chunk_type = item.get("type", "text")

            embedding = embedding_model.embed_by_type(text, chunk_type=chunk_type)

            results.append({
                "index": i,
                "embedding": embedding.tolist(),
                "type": chunk_type
            })

        logger.info(f"✓ Processed {len(results)} items")

        return {
            "object": "list",
            "data": results,
            "model": "ensemble",
            "usage": {
                "total_items": len(results)
            }
        }

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "ensemble",
                "object": "model",
                "owned_by": "local",
                "permission": [],
                "root": "ensemble",
                "parent": None,
                "description": "Ensemble of SPECTER2 + E5-Large + SciBERT",
                "embedding_dimension": 1024
            }
        ]
    }


@app.get("/v1/info")
async def model_info():
    """Get detailed model information"""
    return {
        "name": "EmbedRAG Ensemble",
        "version": "1.0.0",
        "models": [
            {
                "name": "SPECTER2",
                "dimension": 768,
                "purpose": "Scientific papers"
            },
            {
                "name": "E5-Large",
                "dimension": 1024,
                "purpose": "Multilingual"
            },
            {
                "name": "SciBERT",
                "dimension": 768,
                "purpose": "Scientific vocabulary"
            }
        ],
        "final_embedding": {
            "dimension": 1024,
            "normalization": "L2"
        }
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run embedding server"""

    host = os.getenv("EMBEDDING_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("EMBEDDING_SERVER_PORT", "8000"))
    reload = os.getenv("RELOAD", "False").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")
    logger.info("API Documentation available at http://localhost:8000/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
