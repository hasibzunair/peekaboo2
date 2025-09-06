#!/bin/sh

# Script to format codebase with Ruff

# pip install ruff
ruff check . --fix
ruff format .

# To run this file
# chmod +x format.sh
# ./format.sh