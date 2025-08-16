#!/bin/sh

# Script to format codebase

# pip install autopep8
# pip install --force-reinstall --upgrade typed-ast black

# Run autopep8 to fix specific PEP 8 issues
autopep8 --in-place --recursive --select=E1,E2,E3,W1,W2 ./**.py

# Run black to enforce consistent formatting
black ./

# To run this file
# chmod +x format_codebase.sh
# ./format_codebase.sh