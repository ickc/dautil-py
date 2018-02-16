SHELL = /usr/bin/env bash

python = python

test:
	$(python) -m pytest -vv --cov=dautil tests
