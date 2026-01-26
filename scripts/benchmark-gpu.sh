#!/bin/bash

set -ex

uv run benchmark -d gpu -m bartz
uv run benchmark -d gpu -m bartz -t
uv run benchmark -d gpu -m bartz -p
uv run benchmark -d gpu -m bartz -t -p
uv run benchmark -d gpu -m xgboost
uv run benchmark -d gpu -m xgboost -t
uv run benchmark -d gpu -m xgboost -p
uv run benchmark -d gpu -m xgboost -t -p
