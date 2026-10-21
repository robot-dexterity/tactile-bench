#!/usr/bin/env bash
set -euo pipefail

# Work around pybullet's bundled zlib fdopen macro clash on macOS arm64.
export CFLAGS="-Dfdopen=fdopen ${CFLAGS:-}"

exec uv "$@"
