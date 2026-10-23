#!/usr/bin/env bash
# VS Code launches this wrapper so the project depot also applies to its REPL.
THIS_LASSO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export JULIA_DEPOT_PATH="$THIS_LASSO_ROOT/.julia-depot-r0${JULIA_DEPOT_PATH:+:$JULIA_DEPOT_PATH}"
export JULIA_NUM_PRECOMPILE_TASKS=2
export OPENBLAS_NUM_THREADS=1
exec /home/yg/.local/opt/julia-1.9.4/bin/julia "$@"
