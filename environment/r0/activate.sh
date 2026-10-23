# Source this file: source /home/yg/THIS-Lasso/environment/r0/activate.sh
THIS_LASSO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/yg/miniconda3/etc/profile.d/conda.sh
conda activate "$THIS_LASSO_ROOT/.venv-r0"
export PATH="/home/yg/.local/opt/julia-1.9.4/bin:$PATH"
export JULIA_PROJECT="$THIS_LASSO_ROOT/environment/r0/julia"
export JULIA_DEPOT_PATH="$THIS_LASSO_ROOT/.julia-depot-r0"
export JULIA_NUM_PRECOMPILE_TASKS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
