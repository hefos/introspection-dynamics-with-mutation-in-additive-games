#!/usr/bin/env bash
# Build the marked-up ("all changes highlighted") copy that the journal requests
# as the revised manuscript with changes shown. The current source is compared
# against its version on the second-review-for-dga branch (the previously
# submitted version).
set -euo pipefail

baseline="second-review-for-dga"

build_diff() {
  local old_source="$1" new_source="$2" stem="$3"
  git show "${baseline}:${old_source}" > "/tmp/${stem}_old.tex"
  latexdiff --allow-spaces "/tmp/${stem}_old.tex" "${new_source}" \
    > "${stem}-diff.tex"
  latexmk -xelatex -interaction=nonstopmode "${stem}-diff.tex" > /dev/null
  echo "wrote ${stem}-diff.pdf"
}

build_diff "tex/main.tex" "main.tex" "main"
