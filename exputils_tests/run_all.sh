# Run all tests

# Change to the root directory
echo "cwd: $(pwd)"

# https://stackoverflow.com/questions/74734191/how-to-set-the-environment-variable-jupyter-platform-dirs-1
export JUPYTER_PLATFORM_DIRS=1

# check if it is the root directory
if [ ! -f "exputils_tests/run_all.sh" ]; then
    echo "Error: This script must be run from the root directory"
    exit 1
fi

# Run tests
echo "Running all tests..."
pytest -v exputils_tests