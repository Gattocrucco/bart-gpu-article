CUDA_VERSION = $(shell nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | cut -d' ' -f3)
EXTRAS = $(if $(filter 12 13,$(CUDA_VERSION)),--extra=cuda$(CUDA_VERSION),)
UV_RUN = uv run $(EXTRAS)
UV_RUN_NO_R = RPY2_CFFI_MODE=ABI $(UV_RUN)

.PHONY: help
help:
	@echo "Available targets:"
	@echo setup
	@echo benchmark-cpu
	@echo benchmark-gpu
	@echo test-rmse

.PHONY: setup
setup:
	$(UV_RUN_NO_R) python -c 'print("hello there!")'

.PHONY: benchmark-cpu
benchmark-cpu:
	$(UV_RUN_NO_R) benchmark -d cpu -m bartz
	$(UV_RUN_NO_R) benchmark -d cpu -m bartz -t
	$(UV_RUN_NO_R) benchmark -d cpu -m bartz -p
	$(UV_RUN_NO_R) benchmark -d cpu -m bartz -t -p
	$(UV_RUN_NO_R) benchmark -d cpu -m xgboost
	$(UV_RUN_NO_R) benchmark -d cpu -m xgboost -t
	$(UV_RUN_NO_R) benchmark -d cpu -m xgboost -p
	$(UV_RUN_NO_R) benchmark -d cpu -m xgboost -t -p
	$(UV_RUN) benchmark -d cpu -m dbarts
	$(UV_RUN) benchmark -d cpu -m dbarts -t
	$(UV_RUN) benchmark -d cpu -m dbarts -p
	$(UV_RUN) benchmark -d cpu -m dbarts -t -p

.PHONY: benchmark-gpu
benchmark-gpu:
	$(UV_RUN_NO_R) benchmark -d gpu -m bartz
	$(UV_RUN_NO_R) benchmark -d gpu -m bartz -t
	$(UV_RUN_NO_R) benchmark -d gpu -m bartz -p
	$(UV_RUN_NO_R) benchmark -d gpu -m bartz -t -p
	$(UV_RUN_NO_R) benchmark -d gpu -m xgboost
	$(UV_RUN_NO_R) benchmark -d gpu -m xgboost -t
	$(UV_RUN_NO_R) benchmark -d gpu -m xgboost -p
	$(UV_RUN_NO_R) benchmark -d gpu -m xgboost -t -p

.PHONY: test-rmse
test-rmse:
	$(UV_RUN) test-rmse
	$(UV_RUN) test-rmse -t
	$(UV_RUN) test-rmse -p
	$(UV_RUN) test-rmse -t -p
