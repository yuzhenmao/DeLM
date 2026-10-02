#!/bin/sh
exec env LD_LIBRARY_PATH="$CONTEXT_LIBRARY_PATH" "$CONTEXT_PYTHON" /tools/context_client.py "$@"
