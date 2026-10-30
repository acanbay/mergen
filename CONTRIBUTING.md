# Contributing to mergen

Thank you for your interest in contributing to mergen.
All contributions are welcome — bug reports, feature requests, documentation improvements, and code.

---

## Reporting Bugs

Please open an issue on [GitHub](https://github.com/acanbay/mergen/issues) with:

- A minimal reproducible example
- Your Python version and operating system
- The full error traceback

---

## Requesting Features

Open an issue describing:

- The use case you have in mind
- What you would expect the API to look like
- Why existing functionality does not cover it

---

## Contributing Code

### Setup

```bash
git clone https://github.com/acanbay/mergen.git
cd mergen
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a pull request.

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for all public functions
- Write numpy-style docstrings for all public functions and classes
- Include a `References` section in docstrings where applicable

### Pull Request Process

1. Fork the repository and create a branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass: `pytest tests/ -v`
4. Update `CHANGELOG.md` under an `[Unreleased]` section
5. Open a pull request with a clear description of the changes

### Releasing (maintainers)

1. On a branch, set the version in `src/mergen/__init__.py`, move the
   `CHANGELOG.md` entries from `[Unreleased]` to the new version with
   the release date, and update `version` and `date-released` in
   `CITATION.cff`. Merge it into `main` through a pull request.
2. Wait until the `docs-gallery` workflow for that merge has finished
   successfully: it rebuilds the example gallery from the new code,
   and the documentation of a release is built from that gallery.
3. Publish a GitHub release from the latest commit of `main`, with the
   tag `vX.Y.Z` matching the version. This uploads the package to PyPI
   (`publish.yml`), archives the release on Zenodo, and Read the Docs
   builds the documentation of the new version.
4. When the conda-forge bot opens its pull request on the
   `mergen-doe-feedstock`, check that the dependencies in its recipe
   match `pyproject.toml`, then merge it.

---

## Scientific Standards

mergen aims to be publication-quality software. For any new algorithm or metric:

- Provide a peer-reviewed reference
- Include the mathematical formulation in the docstring
- Add a corresponding test that verifies correctness against the reference

---

## AI assistance

Anthropic's Claude (via Claude Code) was used during development for
code auditing, debugging, code and test generation, and documentation
drafting. All AI-assisted output was reviewed, tested and validated by
the maintainers, who take full responsibility for the code.

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
