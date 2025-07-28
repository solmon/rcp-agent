#!/bin/bash

echo "🍳 Setting up Recipe Agent UI..."
echo "================================"

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
    echo ""
    echo "🚀 You can now start the UI server with:"
    echo "   uv run poe ui"
    echo "   OR"
    echo "   uv run python ui_main.py"
    echo ""
    echo "🌐 The UI will be available at: http://localhost:8000"
    echo ""
    echo "📖 For more details, check UI_README.md"
else
    echo "❌ Failed to install dependencies"
    echo "Please check the error messages above"
    exit 1
fi
