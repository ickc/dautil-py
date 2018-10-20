SHELL = /usr/bin/env bash

python = python

.PHONY: all clean test pypi pypiManual pep8 pep8Strict pyflakes flake8 pylint autopep8 autopep8Aggressive past

all:
	python -m compileall .

clean:
	rm -f .coverage docs/dautil*.rst docs/modules.rst docs/README.rst
	rm -rf htmlcov *.egg-info docs/_build .cache dist build
	find -type f -name "*.py[co]" -delete -or -type d -name "__pycache__" -delete

test:
	$(python) -m pytest -vv --cov=dautil tests

docs/%.rst: %.md
	pandoc -s -o $@ $<

# TODO set version
.PHONY: docs
docs: docs/README.rst
	sphinx-apidoc -d 10 -f -e -o $@ . tests
	cd $@ && make html

# Deploy to PyPI
## by Travis, properly git tagged
pypi:
	git tag -a v$$($(python) setup.py --version) -m 'Deploy to PyPI' && git push origin v$$($(python) setup.py --version)
## Manually
pypiManual: docs
	$(python) setup.py sdist bdist_wheel
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

# check python styles
pep8:
	pycodestyle . --ignore=E402,E501,E731
pep8Strict:
	pycodestyle .
pyflakes:
	pyflakes .
flake8:
	flake8 .
pylint:
	pylint dautil

# cleanup python
autopep8:
	autopep8 . --recursive --in-place --pep8-passes 2000 --verbose --ignore=E402,E501,E731
autopep8Aggressive:
	autopep8 . --recursive --in-place --pep8-passes 2000 --verbose --aggressive --aggressive

# pasteurize
past:
	pasteurize -wnj 4 .
