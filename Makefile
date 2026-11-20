.PHONY: test smoke train format lint
ENV = source scripts/env.sh
SHELL := /bin/bash

test:
	$(ENV); "$$NEXARM_PYTHON" -m unittest discover -s tests -v

smoke:
	$(ENV); NEXARM_TEST_DEVICE=cuda PYTHONPATH=tests "$$NEXARM_PYTHON" tests/smoke_act.py

train:
	bash scripts/train.sh --config configs/act_5hz_gpu.yaml

format:
	$(ENV); "$$NEXARM_PYTHON" -m ruff format imitation_lab scripts tests

lint:
	$(ENV); "$$NEXARM_PYTHON" -m ruff check imitation_lab scripts tests
