#!/bin/bash
set -e

# If the release binary does not exist yet (e.g., initial volume mount)
if [ ! -f "/app/rust_optimizer/target/release/christmas_tree_optimizer" ]; then
    echo "🔨 Building Rust release binary inside container..."
    cd /app/rust_optimizer && cargo build --release
    cd /app
fi

exec "$@"
