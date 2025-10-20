#!/bin/bash
# Docker PostgreSQL + pgvector startup script

echo "🐳 Starting Docker containers..."
echo "Platform: Apple Silicon (arm64)"

# Create data directories if they don't exist
mkdir -p data/postgres
mkdir -p data/pgvector

echo "✓ Data directories created"

# Start Docker containers
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if all containers are running
if docker ps | grep -q embed_rag_pgvector && docker ps | grep -q embed_rag_postgres; then
    echo "✓ Containers started successfully"
    echo ""
    echo "📊 Service Status:"
    docker-compose ps
    echo ""

    # Check embedding server
    if docker ps | grep -q embed_rag_embeddings; then
        echo "✅ Embedding Server is starting (may take 1-2 min to download models)"
        echo "   - Wait for: Container state = 'Up'"
        echo "   - Check status: curl http://localhost:8000/health"
    fi

    echo ""
    echo "✅ Infrastructure ready!"
    echo ""
    echo "📚 Connection Details:"
    echo ""
    echo "  PostgreSQL:"
    echo "    postgresql://postgres:postgres@localhost:5432/embed_rag"
    echo ""
    echo "  Embedding Server (OpenAI-compatible):"
    echo "    http://localhost:8000"
    echo "    API Docs: http://localhost:8000/docs"
    echo ""
    echo "🔧 Useful Commands:"
    echo ""
    echo "  View logs:"
    echo "    docker-compose logs -f postgres"
    echo "    docker-compose logs -f embeddings"
    echo ""
    echo "  Test embedding API:"
    echo "    curl -X POST http://localhost:8000/v1/embeddings \\"
    echo "      -H 'Content-Type: application/json' \\"
    echo "      -d '{\"input\": \"hello world\"}'"
    echo ""
    echo "  Test database:"
    echo "    psql postgresql://postgres:postgres@localhost:5432/embed_rag"
    echo ""
    echo "  Stop containers:"
    echo "    docker-compose down"
    echo ""
else
    echo "❌ Container startup failed"
    docker-compose logs
    exit 1
fi
