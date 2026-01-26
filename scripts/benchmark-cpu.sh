#!/bin/bash

set -ex

uv run benchmark -d cpu -m bartz
uv run benchmark -d cpu -m bartz -t
uv run benchmark -d cpu -m bartz -p
uv run benchmark -d cpu -m bartz -t -p
uv run benchmark -d cpu -m dbarts
uv run benchmark -d cpu -m dbarts -t
uv run benchmark -d cpu -m dbarts -p
uv run benchmark -d cpu -m dbarts -t -p
uv run benchmark -d cpu -m xgboost
uv run benchmark -d cpu -m xgboost -t
uv run benchmark -d cpu -m xgboost -p
uv run benchmark -d cpu -m xgboost -t -p
