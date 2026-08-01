# nvidia-smi -q says "CUDA Version : 13.3" up to driver 610, "CUDA UMD Version"
# from 610 on (both until CUDA 14, when the old spelling disappears)
CUDA_VERSION = $(shell nvidia-smi -q 2>/dev/null | grep -o 'CUDA[^:]*Version *: *[0-9]*' | grep -o '[0-9]*' | head -1)
EXTRAS = $(if $(filter 12 13,$(CUDA_VERSION)),--extra=cuda$(CUDA_VERSION),)
UV_RUN = RPY2_CFFI_MODE=ABI uv run $(EXTRAS)

# the simulated high-p dataset for fullbench, produced by `make fullbench-data`;
# keep the directory name in sync with the savedata arguments in that target
SIMDATASET = data/savedata-4000000-1000-32-continuous-peff100

# datasets for fullbench; not data/* because that also matches list-datasets.csv
DATASETS ?= $(wildcard data/dataset-*) $(SIMDATASET)

.PHONY: help
help:
	@echo "Available targets:"
	@echo setup
	@echo benchmark-cpu
	@echo benchmark-gpu
	@echo compare-bart-packages
	@echo fullbench-data
	@echo fullbench-gpu
	@echo article-plots
	@echo copy-plots

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

.PHONY: compare-bart-packages
compare-bart-packages:
	$(UV_RUN) compare-bart-packages
	$(UV_RUN) compare-bart-packages -t
	$(UV_RUN) compare-bart-packages -p
	$(UV_RUN) compare-bart-packages -t -p

.PHONY: fullbench-data
fullbench-data:
	$(UV_RUN) save-datasets
	test -d $(SIMDATASET) || $(UV_RUN) savedata -n 4000000 -p 1000 -q 32 --peff 100 -s 20260801

.PHONY: fullbench-gpu
fullbench-gpu:
	$(UV_RUN) fullbench -d gpu -m bartz $(ARGS) $(DATASETS)
	$(UV_RUN) fullbench -d gpu -m bartz2000 $(ARGS) $(DATASETS)
	$(UV_RUN) fullbench -d gpu -m xgboost $(ARGS) $(DATASETS)

.PHONY: article-plots
article-plots:
	MPLBACKEND=agg $(UV_RUN) benchmark-plot --filter --single-figure
	MPLBACKEND=agg $(UV_RUN) benchmark-plot --filter
	MPLBACKEND=agg $(UV_RUN) benchmark-plot --filter --factor 1000
	MPLBACKEND=agg $(UV_RUN) compare-bart-packages-plot --what mse --single-figure
	MPLBACKEND=agg $(UV_RUN) compare-bart-packages-plot --what mse
	MPLBACKEND=agg $(UV_RUN) compare-bart-packages-plot --what coverage_truth_50

.PHONY: copy-plots
copy-plots:
	cp plots/benchmark-plot.pdf article/time-all.pdf
	cp plots/benchmark-plot-1.pdf article/time-single.pdf
	cp plots/benchmark-plot-1-1000.pdf article/time-single-1000.pdf
	cp plots/compare-bart-packages-plot-mse.pdf article/rmse-all.pdf
	cp plots/compare-bart-packages-plot-mse-1.pdf article/rmse-single.pdf
	cp plots/compare-bart-packages-plot-coverage_truth_50-1.pdf article/coverage-single.pdf
