-- Initialize pgvector extension and database schema

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for embeddings
CREATE SCHEMA IF NOT EXISTS embeddings;

-- Create paper_embeddings table
CREATE TABLE IF NOT EXISTS embeddings.paper_embeddings (
    id VARCHAR PRIMARY KEY,
    paper_id VARCHAR NOT NULL,
    chunk_type VARCHAR NOT NULL CHECK (chunk_type IN ('text', 'formula', 'table', 'theorem')),
    content TEXT,
    latex_content TEXT,
    embedding vector(1024),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_paper_id ON embeddings.paper_embeddings(paper_id);
CREATE INDEX idx_chunk_type ON embeddings.paper_embeddings(chunk_type);
CREATE INDEX idx_created_at ON embeddings.paper_embeddings(created_at DESC);

-- Create IVFFLAT index for vector similarity (cosine distance)
CREATE INDEX idx_embedding_cosine ON embeddings.paper_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create composite index for filtered searches
CREATE INDEX idx_paper_chunk_type ON embeddings.paper_embeddings(paper_id, chunk_type);

-- Create metadata index for JSONB queries
CREATE INDEX idx_metadata ON embeddings.paper_embeddings USING GIN(metadata);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA embeddings TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA embeddings TO postgres;

-- Log initialization
SELECT 'pgvector initialized successfully' as status;
