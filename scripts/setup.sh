#!/bin/bash
# Setup script for Yuzu Memory Builder

set -e

echo "🌸 Yuzu Memory Builder Setup"
echo "============================"

# Check Python version
python3 --version || { echo "❌ Python 3 not found"; exit 1; }

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download model (optional, will also download on first run)
read -p "Download embedding model now? (1.5GB, takes ~2 min) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Downloading model..."
    python3 -c "
from src.core.onnx_server import ONNXServer
onnx = ONNXServer()
onnx.download_model()
"
    echo "✅ Model downloaded"
fi

# Create .env from template
if [ ! -f .env ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your actual credentials!"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your credentials"
echo "2. Place your yuzu_core.db in this directory"
echo "3. Run: python3 main.py"
