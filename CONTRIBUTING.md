# Contributing to Anomaly Detection System

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/anomaly-detection-system.git
cd anomaly-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

- We use [Black](https://github.com/psf/black) for Python code formatting
- We use [Flake8](https://flake8.pycqa.org/) for linting
- We use [mypy](http://mypy-lang.org/) for type checking

Run these before committing:

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/
```

### Testing

We use `pytest` for testing. Please add tests for any new features.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_lstm_autoencoder.py
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Example:
```
Add LSTM bidirectional support

- Implement bidirectional parameter in LSTMAutoencoder
- Update tests for bidirectional mode
- Add documentation for new parameter

Fixes #123
```

## Bug Reports

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

We track feature requests through GitHub Issues. When requesting a feature:

1. **Check existing issues** - your idea might already be discussed
2. **Provide context** - explain why this feature would be useful
3. **Be specific** - provide detailed requirements and use cases
4. **Consider alternatives** - mention other solutions you've considered

## Code Review Process

The core team looks at Pull Requests on a regular basis. After feedback has been given, we expect responses within two weeks. After two weeks, we may close the PR if it isn't showing any activity.

## Community

- Be respectful and inclusive
- Welcome newcomers
- Help others learn
- Share knowledge
- Give constructive feedback

## Attribution

This Contributing Guide is adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/master/CONTRIBUTING.md).

## Questions?

Feel free to open an issue with the `question` label or reach out to the maintainers directly.

Thank you for contributing! 🎉
