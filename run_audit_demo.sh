#!/usr/bin/env bash
# Runs every audit-experiment analysis (Studies 3 and 5) on the data shipped in this repo.
# Regression tables are written to <script>.log next to each script; figures are saved as PDFs.
set -euo pipefail
export MPLBACKEND=Agg
cd "$(dirname "$0")/audit_experiments"
for d in paywalled_papers_audit dataset_audit; do
  for f in regression_and_figures.py analysis_bonferroni.py; do
    echo "Running $d/$f"
    (cd "$d" && python "$f" > "${f%.py}.log" 2> /dev/null)
  done
done
echo "Done. Outputs are in audit_experiments/*/ (*.log, *.pdf, *.csv)."
