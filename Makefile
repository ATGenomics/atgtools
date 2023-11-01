.PHONY: all lint test install dev clean distclean

PYTHON ?= python
PREFIX ?= $(CONDA_PREFIX)

all: ;

lint:
	mypy
	flake8

test: all
	ATGTEST= pytest

test_atg: all
	MYSTERY_STEW= pytest -k test_atg -n auto

install: all
	$(PYTHON) setup.py install && \
	mkdir -p $(PREFIX)/etc/conda/activate.d && \
	cp bin/activate_atg_tab_completion.sh $(PREFIX)/etc/conda/activate.d/

.PHONY: docs
docs:
	# Generate docs - typer-cli must be referenced to the venv installation
	typer atg utils docs --name atgtools --output docs/USAGE.md

.PHONY: build
build:
	# Build the project
	poetry build

.PHONY: publish
publish:
	# Publish to PyPi
	poetry publish --build