CUDA_VERSION = $(shell nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | cut -d' ' -f3)
EXTRAS = $(if $(filter 12 13,$(CUDA_VERSION)),--extra=cuda$(CUDA_VERSION),)
UV_RUN_R = uv run $(EXTRAS)
UV_RUN = RPY2_CFFI_MODE=ABI $(UV_RUN_R)

.PHONY: help
help:
	@echo "Available targets:"
	@echo setup
	@echo benchmark-cpu
	@echo benchmark-gpu
	@echo test-rmse

.PHONY: setup
setup:
	$(UV_RUN) python -c 'print("hello there!")'

.PHONY: benchmark-cpu
benchmark-cpu:
	$(UV_RUN) benchmark -d cpu -m bartz
	$(UV_RUN) benchmark -d cpu -m bartz -t
	$(UV_RUN) benchmark -d cpu -m bartz -p
	$(UV_RUN) benchmark -d cpu -m bartz -t -p
	$(UV_RUN) benchmark -d cpu -m xgboost
	$(UV_RUN) benchmark -d cpu -m xgboost -t
	$(UV_RUN) benchmark -d cpu -m xgboost -p
	$(UV_RUN) benchmark -d cpu -m xgboost -t -p
	$(UV_RUN) benchmark -d cpu -m catboost
	$(UV_RUN) benchmark -d cpu -m catboost -t
	$(UV_RUN) benchmark -d cpu -m catboost -p
	$(UV_RUN) benchmark -d cpu -m catboost -t -p
	$(UV_RUN_R) benchmark -d cpu -m dbarts
	$(UV_RUN_R) benchmark -d cpu -m dbarts -t
	$(UV_RUN_R) benchmark -d cpu -m dbarts -p
	$(UV_RUN_R) benchmark -d cpu -m dbarts -t -p

.PHONY: benchmark-gpu
benchmark-gpu:
	$(UV_RUN) benchmark -d gpu -m bartz
	$(UV_RUN) benchmark -d gpu -m bartz -t
	$(UV_RUN) benchmark -d gpu -m bartz -p
	$(UV_RUN) benchmark -d gpu -m bartz -t -p
	$(UV_RUN) benchmark -d gpu -m xgboost
	$(UV_RUN) benchmark -d gpu -m xgboost -t
	$(UV_RUN) benchmark -d gpu -m xgboost -p
	$(UV_RUN) benchmark -d gpu -m xgboost -t -p
	$(UV_RUN) benchmark -d gpu -m catboost
	$(UV_RUN) benchmark -d gpu -m catboost -t
	$(UV_RUN) benchmark -d gpu -m catboost -p
	$(UV_RUN) benchmark -d gpu -m catboost -t -p

.PHONY: test-rmse
test-rmse:
	$(UV_RUN_R) test-rmse
	$(UV_RUN_R) test-rmse -t
	$(UV_RUN_R) test-rmse -p
	$(UV_RUN_R) test-rmse -t -p
