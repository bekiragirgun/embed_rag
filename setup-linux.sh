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

# Check Python 3.9+
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not installed${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION installed"

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

# Create Python virtual environment
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate and install dependencies
source venv/bin/activate
echo "Installing Python dependencies..."
pip install --upgrade pip > /dev/null
pip install -r requirements.txt
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
echo "  3. Run processing: python3 main_pipeline.py"
echo ""
echo "To activate virtual environment:"
echo "  source venv/bin/activate"
echo ""
