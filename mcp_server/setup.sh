#!/bin/bash
# Setup script for MCP Context Manager

set -e

echo "🚀 MCP Context Manager Setup"
echo "============================"
echo ""

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "📁 Installation directory: $SCRIPT_DIR"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r "$SCRIPT_DIR/requirements.txt"
echo "✅ Dependencies installed"
echo ""

# Create contexts directory
CONTEXTS_DIR="$HOME/.mcp_contexts"
if [ ! -d "$CONTEXTS_DIR" ]; then
    echo "📂 Creating contexts directory: $CONTEXTS_DIR"
    mkdir -p "$CONTEXTS_DIR"
    echo "✅ Contexts directory created"
else
    echo "✅ Contexts directory exists: $CONTEXTS_DIR"
fi
echo ""

# Make server.py executable
chmod +x "$SCRIPT_DIR/server.py"

# Generate configuration
echo "⚙️  Generating configuration..."
CONFIG_FILE="$SCRIPT_DIR/claude-config.json"

cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "context-manager": {
      "command": "$SCRIPT_DIR/venv/bin/python",
      "args": [
        "$SCRIPT_DIR/server.py"
      ],
      "env": {
        "CONTEXT_STORAGE_DIR": "$CONTEXTS_DIR",
        "PYTHONPATH": "$SCRIPT_DIR"
      }
    }
  }
}
EOF

echo "✅ Configuration generated: $CONFIG_FILE"
echo ""

# Print instructions
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo ""
echo "1️⃣  Add to Claude Desktop config (~/.config/claude/config.json):"
echo ""
echo "   Copy the content from: $CONFIG_FILE"
echo ""
echo "   Or manually add:"
echo "   {"
echo "     \"mcpServers\": {"
echo "       \"context-manager\": {"
echo "         \"command\": \"$SCRIPT_DIR/venv/bin/python\","
echo "         \"args\": [\"$SCRIPT_DIR/server.py\"],"
echo "         \"env\": {"
echo "           \"CONTEXT_STORAGE_DIR\": \"$CONTEXTS_DIR\","
echo "           \"PYTHONPATH\": \"$SCRIPT_DIR\""
echo "         }"
echo "       }"
echo "     }"
echo "   }"
echo ""
echo "2️⃣  Restart Claude Desktop"
echo ""
echo "3️⃣  Test with: ls ~/.mcp_contexts"
echo ""
echo "📚 For more information, see README.md"
echo ""
