# Agent Guidelines for SQD_Agent

This document outlines the conventions and commands for agents operating within this repository.

## 1. Build/Lint/Test Commands

- **Install Dependencies**: `pip install -r requirements.txt`
- **Run All Tests**: `pytest` (assuming pytest is installed and configured)
- **Run a Single Test**: `pytest <path_to_test_file>::<test_function_name>`
- **Linting**: `flake8 .` (assuming flake8 is installed and configured)
- **Type Checking**: `mypy .` (if mypy is used)

## 2. Code Style Guidelines

- **Imports**: Organize imports alphabetically, with standard library imports first, followed by third-party, and then local imports.
- **Formatting**: Adhere to PEP 8. Use a linter like `flake8` and a formatter like `black` for consistent code style.
- **Types**: Use type hints for function arguments and return values.
- **Naming Conventions**: Follow PEP 8 for naming (e.g., `snake_case` for functions and variables, `CamelCase` for classes).
- **Error Handling**: Use specific exception types. Avoid broad `except` clauses. Log errors appropriately.