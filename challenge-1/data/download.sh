#!/bin/bash

# Exit on error
set -e

# Competition slug
KAGGLE_COMPETITION="soil-classification"
TARGET_DIR="./data"

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ 'kaggle' command not found. Install it with: pip install kaggle"
    exit 1
fi

# Create data directory
mkdir -p "$TARGET_DIR"

# Download competition data
echo "📦 Downloading competition: $KAGGLE_COMPETITION"
kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$TARGET_DIR"

echo "✅ Download complete. Files saved to $TARGET_DIR"
