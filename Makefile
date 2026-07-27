# nvidia-smi -q says "CUDA Version : 13.3" up to driver 610, "CUDA UMD Version"
# from 610 on (both until CUDA 14, when the old spelling disappears)
CUDA_VERSION = $(shell nvidia-smi -q 2>/dev/null | grep -o 'CUDA[^:]*Version *: *[0-9]*' | grep -o '[0-9]*' | head -1)
EXTRAS = $(if $(filter 12 13,$(CUDA_VERSION)),--extra=cuda$(CUDA_VERSION),)
UV_RUN = RPY2_CFFI_MODE=ABI uv run $(EXTRAS)

# datasets for fullbench; not data/* because that also matches list-datasets.csv
DATASETS ?= $(wildcard data/dataset-*)

.PHONY: help
help:
	@echo "Available targets:"
	@echo setup
	@echo benchmark-cpu
	@echo benchmark-gpu
	@echo test-rmse
	@echo fullbench-cpu
	@echo fullbench-gpu

.PHONY: setup
setup:
	$(UV_RUN) python -c 'print("hello there!")'

.PHONY: benchmark-matrix
benchmark-matrix:
	$(UV_RUN) benchmark $(ARGS)
	$(UV_RUN) benchmark $(ARGS) -t
	$(UV_RUN) benchmark $(ARGS) -p
	$(UV_RUN) benchmark $(ARGS) -t -p

.PHONY: benchmark-cpu
benchmark-cpu:
	$(MAKE) benchmark-matrix ARGS='-d cpu -m bartz'
	$(MAKE) benchmark-matrix ARGS='-d cpu -m xgboost'
	$(MAKE) benchmark-matrix ARGS='-d cpu -m catboost'
	$(MAKE) benchmark-matrix ARGS='-d cpu -m dbarts'

.PHONY: benchmark-gpu
benchmark-gpu:
	$(MAKE) benchmark-matrix ARGS='-d gpu -m bartz'
	$(MAKE) benchmark-matrix ARGS='-d gpu -m xgboost'
	$(MAKE) benchmark-matrix ARGS='-d gpu -m catboost'

.PHONY: test-rmse
test-rmse:
	$(UV_RUN) test-rmse
	$(UV_RUN) test-rmse -t
	$(UV_RUN) test-rmse -p
	$(UV_RUN) test-rmse -t -p

.PHONY: fullbench-cpu
fullbench-cpu:
	$(UV_RUN) fullbench -d cpu -m bartz $(ARGS) $(DATASETS)
	$(UV_RUN) fullbench -d cpu -m bartzadaptive $(ARGS) $(DATASETS)
	$(UV_RUN) fullbench -d cpu -m xgboost $(ARGS) $(DATASETS)

.PHONY: fullbench-gpu
fullbench-gpu:
	$(UV_RUN) fullbench -d gpu -m bartz $(ARGS) $(DATASETS)
	$(UV_RUN) fullbench -d gpu -m bartzadaptive $(ARGS) $(DATASETS)
	$(UV_RUN) fullbench -d gpu -m xgboost $(ARGS) $(DATASETS)

.PHONY: copy-plots
copy-plots:
	cp plots/test-rmse-plot.pdf article/rmse-all.pdf
	cp plots/test-rmse-plot-1.pdf article/rmse-single.pdf
	cp plots/benchmark-plot.pdf article/time-all.pdf
	cp plots/benchmark-plot-1.pdf article/time-single.pdf
