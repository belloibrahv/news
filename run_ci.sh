#!/bin/bash
set -e

# News Article Classification System — Local CI/CD Validation Script
echo "=========================================================================="
echo "STARTING LOCAL CI/CD RUNNER"
echo "=========================================================================="

# 1. Check virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
else
    echo "Virtual environment found."
fi

# 2. Download NLTK data
echo "Ensuring NLTK packages are up to date..."
.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# 3. Run unit tests
echo ""
echo "--------------------------------------------------------------------------"
echo "RUNNING AUTOMATED UNIT TESTS"
echo "--------------------------------------------------------------------------"
.venv/bin/python -m unittest src/test_suite.py

# 4. Run system tests
echo ""
echo "--------------------------------------------------------------------------"
echo "RUNNING FUNCTIONAL SYSTEM TESTS"
echo "--------------------------------------------------------------------------"
.venv/bin/python src/system_test.py

echo "=========================================================================="
echo "CI/CD BUILD STATUS: SUCCESSFUL (ALL TESTS PASSED)"
echo "=========================================================================="
exit 0
