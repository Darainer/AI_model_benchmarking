# Shortcuts for the Dockerized Jetson benchmark.
# Usage: make build | make gpu-check | make download | make bench | ...

COMPOSE := docker compose
SVC     := benchmark
RUN     := $(COMPOSE) run --rm $(SVC)

.PHONY: build pull gpu-check shell download bench bench-cpu bench-file clean venv

## Host venv setup (Jetson) — inherits the JetPack GPU stack, then installs the
## shared deps via scripts/install_deps.sh (the same script the Docker build
## uses). Use this path for HW-decoded video/camera runs. See VENV.md.
venv:
	bash scripts/setup_venv.sh

## Build the image (run `make pull` first for the one-time base download)
build:
	$(COMPOSE) build

## Pre-pull the large dustynv base image (one-time, several GB)
pull:
	docker pull dustynv/l4t-ml:r36.4.0

## Sanity check: confirm onnxruntime sees the GPU execution providers
gpu-check:
	$(RUN) python3 -c "import onnxruntime as ort; print('providers:', ort.get_available_providers())"

## Interactive shell inside the container
shell:
	$(RUN) bash

## Download / export all model checkpoints into ./models
download:
	$(RUN) python3 scripts/download_models.py

## Benchmark on synthetic frames (pure inference timing)
bench:
	$(RUN) python3 scripts/run_benchmark.py

## Benchmark forcing CPU only (no GPU needed)
bench-cpu:
	$(RUN) python3 scripts/run_benchmark.py --providers CPUExecutionProvider

## Benchmark a real video file (host path):  make bench-file VIDEO=/home/user/clip.mp4
bench-file:
	$(COMPOSE) run --rm \
	  -v $(dir $(abspath $(VIDEO))):/video:ro \
	  $(SVC) python3 scripts/run_benchmark.py --input-type file --source /video/$(notdir $(VIDEO))

## Remove the built image
clean:
	docker image rm ai-model-benchmarking:orin 2>/dev/null || true
