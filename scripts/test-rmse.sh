#!/bin/bash

set -ex

uv run test-rmse
uv run test-rmse -t
uv run test-rmse -p
uv run test-rmse -t -p
