#!/usr/bin/env bash
#
# Compile paper.tex to PDF and clean up LaTeX artifacts.
#
# Usage:
#   ./build.sh            # compile paper.tex
#   ./build.sh myfile     # compile myfile.tex (omit the .tex extension)
#   ./build.sh -c         # only clean artifacts, do not compile
#
set -euo pipefail

# Run from the directory that holds this script, so relative paths
# (media/, references.bib, PRIMEarxiv.sty) always resolve.
cd "$(dirname "$0")"

JOB="paper"
CLEAN_ONLY=0
if [[ "${1:-}" == "-c" ]]; then
  CLEAN_ONLY=1
elif [[ -n "${1:-}" ]]; then
  JOB="${1%.tex}"          # strip a trailing .tex if the user passed one
fi

# Artifacts produced by pdflatex/bibtex that we remove afterwards.
ARTIFACTS=(aux bbl blg log out toc lof lot fls fdb_latexmk synctex.gz nav snm vrb)

clean() {
  for ext in "${ARTIFACTS[@]}"; do
    rm -f "${JOB}.${ext}"
  done
  rm -f missfont.log texput.log
}

if [[ $CLEAN_ONLY -eq 1 ]]; then
  echo ">> Cleaning artifacts for ${JOB}"
  clean
  echo ">> Done."
  exit 0
fi

if [[ ! -f "${JOB}.tex" ]]; then
  echo "error: ${JOB}.tex not found in $(pwd)" >&2
  exit 1
fi

echo ">> Compiling ${JOB}.tex"

if command -v latexmk >/dev/null 2>&1; then
  # latexmk runs pdflatex + bibtex as many times as needed.
  latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error "${JOB}.tex"
else
  # Manual sequence: pdflatex, bibtex, then two more pdflatex passes so
  # citations and cross-references resolve.
  pdflatex -interaction=nonstopmode -halt-on-error "${JOB}.tex"
  bibtex   "${JOB}"
  pdflatex -interaction=nonstopmode -halt-on-error "${JOB}.tex"
  pdflatex -interaction=nonstopmode -halt-on-error "${JOB}.tex"
fi

echo ">> Cleaning artifacts"
clean

if [[ -f "${JOB}.pdf" ]]; then
  echo ">> Built ${JOB}.pdf"
else
  echo "error: ${JOB}.pdf was not produced" >&2
  exit 1
fi
