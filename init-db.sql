-- PostgreSQL initialization script
-- Basic setup (pgvector will be added via separate initialization)

CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Logging for debugging
SELECT 'PostgreSQL initialized successfully' as status;
