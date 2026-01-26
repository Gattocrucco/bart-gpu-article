#!/bin/bash

set -e

# uv run benchmark -m bartz
# uv run benchmark -m bartz -t
# uv run benchmark -m bartz -p
# uv run benchmark -m bartz -t -p
# uv run benchmark -m dbarts
# uv run benchmark -m dbarts -t
# uv run benchmark -m dbarts -p
# uv run benchmark -m dbarts -t -p
# uv run benchmark -m xgboost
# uv run benchmark -m xgboost -t
# uv run benchmark -m xgboost -p
# uv run benchmark -m xgboost -t -p

uv run test-rmse
uv run test-rmse -t
uv run test-rmse -p
uv run test-rmse -t -p
