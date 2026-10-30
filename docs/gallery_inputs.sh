#!/usr/bin/env bash
# Prints a fingerprint of the files the executed example gallery is
# built from: the examples, the package source, its dependencies
# (pyproject.toml) and the gallery configuration (docs/conf.py). It
# uses the git tree hashes of the checked-out commit, so the docs-gallery
# workflow and Read the Docs compute the same value for the same code.
set -euo pipefail
git rev-parse HEAD:examples HEAD:src HEAD:pyproject.toml HEAD:docs/conf.py \
    | sha256sum | cut -d' ' -f1
