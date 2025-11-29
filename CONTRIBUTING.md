# Contributing to JSF-Core

Thank you for your interest in contributing to JSF-Core! This document provides guidelines and instructions for contributing.

## 🎯 Ways to Contribute

- **Bug Reports**: Submit detailed bug reports with reproducible examples
- **Feature Requests**: Propose new features or enhancements
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Improve documentation, examples, or tutorials
- **Testing**: Add test cases to improve coverage

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/JBAC-Strategy-Foundry.git
cd JBAC-Strategy-Foundry
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 💻 Development Workflow

### Code Style

We use:
- **Black** for code formatting (100 char line length)
- **Ruff** for linting and import sorting
- **mypy** for type checking

Run formatting and linting:

```bash
make format  # Format code
make lint    # Check linting and types
```

### Testing

Write tests for all new features and bug fixes:

```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_your_module.py -v

# Run specific test function
pytest tests/test_your_module.py::test_function_name -v
```

### Type Hints

All functions must have type hints:

```python
def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns from prices."""
    return prices.pct_change()
```

### Documentation

Use NumPy-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.

    More detailed explanation if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Examples:
        >>> my_function("test", 5)
        True
    """
    pass
```

## 📝 Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): brief description

Detailed explanation if needed.

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- test: Test additions or modifications
- refactor: Code refactoring
- chore: Maintenance tasks
- perf: Performance improvements
```

Examples:
```
feat(signals): add exponential moving average signal
fix(simulation): correct position sizing calculation
docs(readme): update installation instructions
test(strategies): add tests for momentum strategy
```

## 🔍 Pull Request Process

1. **Update Tests**: Ensure all tests pass and add new tests for your changes
2. **Update Documentation**: Update docstrings and README if needed
3. **Run Checks**: Run `make check` to ensure all checks pass
4. **Create PR**: Submit a pull request with a clear description
5. **Code Review**: Address any feedback from reviewers

### PR Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass (`make test`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Commit messages follow conventions
- [ ] No merge conflicts

## 🐛 Reporting Bugs

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Reproduction Steps**: Minimal code to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, jsf-core version
6. **Error Messages**: Full error traceback if applicable

## 💡 Feature Requests

When requesting features:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: How you envision the feature working
3. **Alternatives**: Other approaches you've considered
4. **Examples**: Code examples of how it would be used

## 📖 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Acknowledge different viewpoints and experiences

### Unacceptable Behavior

- Harassment or discrimination of any kind
- Trolling, insulting, or derogatory comments
- Publishing private information without permission

## 📬 Communication

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: contact@jbac.edu for private inquiries

## 🙏 Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes

Thank you for contributing to JSF-Core! 🎉
