.PHONY: quality style pip

PYTHON := python3
BLACK := black
ISORT := isort
FLAKE8 := flake8
AUTOFLAKE := autoflake

# Check that source code meets quality standards
quality:
	$(BLACK) --check --line-length 119 --target-version py38 src/
	$(ISORT) --check-only src/
	$(FLAKE8) --max-line-length 119 src/

# Format source code automatically
style:
	$(AUTOFLAKE) --in-place --remove-all-unused-imports --remove-unused-variables --recursive src/
	$(BLACK) --line-length 119 --target-version py38 src/
	$(ISORT) src/

# Build and publish to PyPI
pip:
	rm -rf build/
	rm -rf dist/
	make style && make quality
	$(PYTHON) -m build
	$(PYTHON) -m twine upload dist/* --verbose
