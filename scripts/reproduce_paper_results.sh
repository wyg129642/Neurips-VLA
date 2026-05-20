#!/usr/bin/env bash
# Reproduce the paper's Tables 2-5 and Figure 3 (no GPU / weights needed).
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
from robogym.analysis import reproduce_paper_tables, reproduce_paper_figures
t = reproduce_paper_tables("results/paper_tables")
f = reproduce_paper_figures("results/figures")
print("tables :", t["report"])
print("figures:", len(f), "files in results/figures")
PY
