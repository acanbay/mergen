#!/usr/bin/env bash
# Read the Docs post_checkout job. Downloads the example gallery that the
# docs-gallery workflow built (branch docs-gallery) into
# docs/auto_examples and checks that it was built from the same
# examples, source and gallery configuration as this commit. If not,
# the build is cancelled (exit code 183) so the pages already online
# stay as they are; the workflow starts the build again once the
# gallery for this code is ready.
set -euo pipefail

want="$(bash docs/gallery_inputs.sh)"
mkdir -p docs/auto_examples
curl -fsSL https://github.com/acanbay/mergen/archive/refs/heads/docs-gallery.tar.gz \
    | tar xz -C docs/auto_examples --strip-components=1
have="$(cat docs/auto_examples/GALLERY_INPUTS 2>/dev/null || echo none)"

if [ "$have" != "$want" ]; then
    echo "The gallery on branch docs-gallery was built from other code"
    echo "(gallery $have, this commit $want). Build cancelled; the"
    echo "docs-gallery workflow starts it again when the gallery is ready."
    exit 183
fi
echo "Gallery matches this commit ($want)."
