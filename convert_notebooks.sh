#!/bin/bash

# Convert all TS*/main.ipynb notebooks to HTML using jupyter nbconvert
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: .venv not found at $SCRIPT_DIR/.venv"
    exit 1
fi

for nb in "$SCRIPT_DIR"/TS*/main.ipynb; do
    if [ -f "$nb" ]; then
        dir=$(dirname "$nb")
        ts_name=$(basename "$dir")
        echo "Converting $ts_name/main.ipynb..."
        "$PYTHON" -m nbconvert --to html --execute "$nb" --output "$dir/main.html"
        if [ $? -eq 0 ]; then
            echo "  -> $ts_name/main.html created successfully"
        else
            echo "  -> ERROR: failed to convert $ts_name/main.ipynb"
        fi
    fi
done

echo "Done."
