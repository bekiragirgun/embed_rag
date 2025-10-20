#!/bin/bash
set -e

echo "=========================================="
echo "RAG System - Ubuntu 25 Setup Script"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: This script is designed for Linux only${NC}"
    exit 1
fi

# Check Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" != "ubuntu" ]]; then
        echo -e "${YELLOW}Warning: Not Ubuntu. Continuing anyway...${NC}"
    fi
fi

echo -e "${GREEN}✓${NC} Running on Linux"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker installed"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker Compose installed"

# Check or install UV
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo -e "${GREEN}✓${NC} UV installed"
else
    echo -e "${GREEN}✓${NC} UV already installed"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/pdfs
mkdir -p data/postgres
mkdir -p data/pgvector
mkdir -p model_cache
mkdir -p logs
echo -e "${GREEN}✓${NC} Directories created"

# Create .env if not exists
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✓${NC} .env created (please update with your settings)"
else
    echo -e "${GREEN}✓${NC} .env already exists"
fi

# Setup Python environment with UV
echo ""
echo "Setting up Python environment with UV..."

# UV will automatically create venv and install dependencies
# Specify Python version if needed: uv venv --python 3.11
if [ ! -d ".venv" ]; then
    uv venv
    echo -e "${GREEN}✓${NC} Virtual environment created with UV"
fi

# Install dependencies with UV (much faster than pip!)
echo "Installing dependencies with UV..."
uv pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"

# Check PDF directory
echo ""
echo "Checking PDF files..."
PDF_COUNT=$(find data/pdfs -name "*.pdf" 2>/dev/null | wc -l)
if [ "$PDF_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}⚠${NC}  No PDF files found in data/pdfs/"
    echo "   Please copy your PDFs to data/pdfs/ before processing"
else
    echo -e "${GREEN}✓${NC} Found $PDF_COUNT PDF files"
fi

# Start Docker services
echo ""
echo "Starting Docker services..."
./docker-up.sh

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Copy PDF files to: data/pdfs/"
echo "  2. Update .env file with your settings"
echo "  3. Activate environment: source .venv/bin/activate"
echo "  4. Run processing: python3 main_pipeline.py"
echo ""
echo "UV Commands:"
echo "  uv pip install <package>  - Install new package"
echo "  uv pip list               - List installed packages"
echo "  uv venv --python 3.11     - Use specific Python version"
echo ""
