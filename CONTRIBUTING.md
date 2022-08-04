# Contributing

When contributing to this repository, please refer to the following.

## Suggested Guidelines

1. When opening a pull request (PR), the title should be clear and concise in describing the changes. The PR description can include a more descriptive log of the changes.
2. If the pull request (PR) is linked to a specific issue, the PR should be linked to the issue. You can use the [Closing Keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) in the PR description to automatically link the issue. Merging a PR will close the linked issue.
3. This repository follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for code formatting.
4. If you are working on improving the speed of *dot*, please read first our guide on [code profiling](docs/profiling.md).

## Setup Dev-Tools

1. Install Dev Requirements

 ```bash
 pip install -r requirements-dev.txt
 ```

2. Install Pre-Commit Hooks

 ```bash
 pre-commit install
 ```

## CI/CD

Run Unit Tests (with coverage):

```bash
pytest --cov=src --cov-report=term-missing:skip-covered --cov-fail-under=10
```

Lock Base and Dev Requirements (pre-requisite: `pip install pip-tools==6.8.0`):

 ```bash
 pip-compile setup.cfg
 pip-compile --extra=dev --output-file=requirements-dev.txt --strip-extras setup.cfg
 ```

## Semantic Versioning

This repository follows the [Semantic Versioning](https://semver.org/) standard.

Bump a major release:

```bash
bumpversion major
```

Bump a minor release:

```bash
bumpversion minor
```

Bump a patch release:

```bash
bumpversion patch
```
