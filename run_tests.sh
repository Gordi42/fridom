#!/bin/bash
# =============================================================================
#  Run tests for the specified backends
# =============================================================================

# Default values:
TARGET="tests/"
BACKENDS=("numpy" "cupy" "jax_cpu" "jax_gpu")

# Function to display usage
usage() {
  echo "Usage: $0 [-t target_path] [-b backends]"
  echo "  -t target_path : Specify the test target path (default: tests/)"
  echo "  -b backends    : Specify the backends as a space-separated string (default: 'numpy cupy jax_cpu jax_gpu')"
  exit 1
}

# Parse command line arguments
while getopts "t:b:" opt; do
  case ${opt} in
    t )
      TARGET=$OPTARG
      ;;
    b )
      BACKENDS=($OPTARG)
      ;;
    \? )
      usage
      ;;
  esac
done
shift $((OPTIND -1))  # Skip parsed options

# Print the configuration
echo "Running tests in $TARGET with backends: ${BACKENDS[@]}"

# first erase old coverage data
coverage erase

# Initialize an error flag
ERROR=0

# Run tests for each backend
for backend in "${BACKENDS[@]}";
do
    echo "Running tests for backend: $backend"
    FRIDOM_BACKEND=$backend coverage run -a -m pytest $TARGET || ERROR=1
done

# Report coverage
coverage report -m

# Exit with error if any test failed
if [ $ERROR -ne 0 ]; then
    echo "Some tests failed."
    exit 1
fi
