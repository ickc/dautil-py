SHELL = /usr/bin/env bash

python = python

.PHONY: clean test pypi pypiManual pep8 pep8Strict pyflakes flake8 pylint autopep8 autopep8Aggressive past

clean:
	rm -f .coverage
	rm -rf htmlcov *.egg-info
	find -type f -name "*.py[co]" -delete -or -type d -name "__pycache__" -delete

test:
	$(python) -m pytest -vv --cov=dautil tests


# Deploy to PyPI
## by Travis, properly git tagged
pypi:
	git tag -a v$$($(python) setup.py --version) -m 'Deploy to PyPI' && git push origin v$$($(python) setup.py --version)
## Manually
pypiManual:
	$(python) setup.py register -r pypitest && $(python) setup.py sdist upload -r pypitest && $(python) setup.py register -r pypi && $(python) setup.py sdist upload -r pypi

# check python styles
pep8:
	pep8 . --ignore=E402,E501,E731
pep8Strict:
	pep8 .
pyflakes:
	pyflakes .
flake8:
	flake8 .
pylint:
	pylint dautil

# cleanup python
autopep8:
	autopep8 . --recursive --in-place --pep8-passes 2000 --verbose
autopep8Aggressive:
	autopep8 . --recursive --in-place --pep8-passes 2000 --verbose --aggressive --aggressive

# pasteurize
past:
	pasteurize -wnj 4 .
