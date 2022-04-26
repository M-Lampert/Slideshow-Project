.PHONY: help install-deps format

.DEFAULT: help
help:
	@echo "install-deps: Installs all dependencies necessary for this project."
	@echo "format: Formats all files in a common format."
	@echo "lint: Check for errors, bugs, stylistic errors and suspicious constructs."

install-deps: requirements.txt
	@pip install -r requirements.txt --upgrade

format:
	@autoflake --remove-all-unused-imports -ir .
	@isort . --overwrite-in-place
	@black . --line-length=160

lint:
	@flake8 --max-line-length=160 .
