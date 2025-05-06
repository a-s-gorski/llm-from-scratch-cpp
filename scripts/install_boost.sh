#!/bin/bash

set -e

BOOST_VERSION=1.84.0
BOOST_VERSION_UNDERSCORED=${BOOST_VERSION//./_}
INSTALL_DIR=$HOME/.local
BOOST_DIR=$INSTALL_DIR/boost_$BOOST_VERSION_UNDERSCORED

if [ -d "$BOOST_DIR" ]; then
    echo "Boost is already installed at $BOOST_DIR"
    exit 0
fi

echo "Downloading Boost $BOOST_VERSION..."
wget -q --show-progress "https://boostorg.jfrog.io/artifactory/main/release/$BOOST_VERSION/source/boost_$BOOST_VERSION_UNDERSCORED.tar.bz2"

echo "Extracting Boost..."
tar --bzip2 -xf "boost_$BOOST_VERSION_UNDERSCORED.tar.bz2"
cd "boost_$BOOST_VERSION_UNDERSCORED"

echo "Bootstrapping Boost..."
./bootstrap.sh --prefix="$BOOST_DIR"

echo "Installing Boost..."
./b2 install -j$(nproc) --prefix="$BOOST_DIR"

echo "Cleaning up..."
cd ..
rm -rf "boost_$BOOST_VERSION_UNDERSCORED" "boost_$BOOST_VERSION_UNDERSCORED.tar.bz2"

echo "Boost installed at $BOOST_DIR"
