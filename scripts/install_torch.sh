#!/bin/bash

set -e

TORCH_VERSION=2.2.0
INSTALL_DIR=$HOME/.local
TORCH_DIR=$INSTALL_DIR/libtorch

if [ -d "$TORCH_DIR" ]; then
    echo "LibTorch is already installed at $TORCH_DIR"
    exit 0
fi

echo "Downloading LibTorch $TORCH_VERSION..."
wget -q --show-progress "https://download.pytorch.org/libtorch/nightly/cu121/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}.dev20230411%2Bcu121.zip"

echo "Extracting LibTorch..."
unzip -q "libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}.dev20230411%2Bcu121.zip" -d $INSTALL_DIR

echo "Cleaning up..."
rm "libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}.dev20230411%2Bcu121.zip"

echo "LibTorch installed at $TORCH_DIR"
