#!/bin/bash
# Submit the benchmark suite as a GPU job on levante.
#
# Usage:
#   ./start_job.sh [extra arguments passed to the benchmark runner]
#
# Examples:
#   ./start_job.sh
#   ./start_job.sh --filter bench_nonhydro --reps 20
#
# Results are written to benchmarks/results/ (see the run output for
# the exact file name). Compare two result files with:
#   uv run python -m fridom.benchmarking compare base.json new.json

REPO_ROOT=$(git rev-parse --show-toplevel)

JOB_SCRIPT=$(mktemp /tmp/fridom_benchmark.XXXXXX.sh)

cat <<EOL > "$JOB_SCRIPT"
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --account=uo0780
#SBATCH --partition=gpu
#SBATCH --job-name=fridom_benchmarks
#SBATCH --output=$REPO_ROOT/benchmarks/results/slurm_%j.out

cd "$REPO_ROOT"
uv run python -m fridom.benchmarking run benchmarks $@
EOL

mkdir -p "$REPO_ROOT/benchmarks/results"
sbatch "$JOB_SCRIPT"
