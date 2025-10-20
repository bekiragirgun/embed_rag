#!/bin/bash
# Quick activation script for embed_rag project

cd "$(dirname "$0")" || exit

echo "🔧 Activating embed_rag environment..."

# Activate venv
source venv/bin/activate

# Load env variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✓ Environment variables loaded"
fi

echo "✓ Ready! Python: $(python --version)"
echo ""
echo "Available commands:"
echo "  python deepdoc_test.py          - Test PDF extraction"
echo "  python ensemble_embeddings.py   - Test embedding models"
echo "  python vector_store.py          - Test vector database"
echo "  python main_pipeline.py         - Run full pipeline"
echo ""
