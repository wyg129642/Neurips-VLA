"""Reported numbers from the paper, used for figure regeneration and checks.

Holds the values reported in Tables 2-5 and Figure 3, so :mod:`robogym.analysis`
can regenerate the Figure 3 radar charts and sanity-check a live runner's CSVs.
For the human-automation alignment study (§4.4) only the reported mean Spearman
``rho ~= 0.9`` (3 experts, 8 models, 5 dimensions) is stored; per-dimension ρ
and the inter-rater Kendall's W are computed by the study harness instead.
"""

METRICS = ["SR", "Comp", "Space", "Time", "Smooth", "Safety"]

# Table 2: Performance on Augmented LIBERO
TABLE2_AUGMENTED_LIBERO = {
    "LingBot-VLA w/ depth": [0.98, 99.34, 94.39, 65.32, 65.21, 69.27],
    "GigaBrain0.1":         [0.98, 99.03, 93.06, 61.01, 67.04, 71.52],
    "X-VLA":                [0.98, 99.37, 91.21, 53.88, 65.34, 54.91],
    "GR00T-N1.6":           [0.97, 98.51, 86.34, 60.24, 64.33, 61.35],
    "DM0":                  [0.95, 96.56, 89.32, 62.54, 66.43, 65.94],
    "Pi0.5":                [0.95, 96.41, 85.12, 56.31, 61.52, 58.92],
    "Pi0":                  [0.94, 94.88, 81.50, 54.77, 56.30, 55.32],
    "RDT":                  [0.94, 95.54, 78.13, 55.23, 43.56, 51.16],
}

# Table 3: Performance on System 2 reasoning tasks
TABLE3_REASONING = {
    "DM0":                  [0.32, 30.92, 14.07, 11.44, 27.84, 32.72],
    "GigaBrain0.1":         [0.28, 32.93, 12.81, 13.43, 26.57, 30.08],
    "LingBot-VLA w/ depth": [0.22, 21.88, 11.55, 11.05, 21.14, 23.83],
    "Pi0.5":                [0.14, 17.54, 10.96, 8.19, 20.48, 11.79],
    "GR00T-N1.6":           [0.12, 14.73, 11.59, 12.19, 24.67, 15.55],
    "Pi0":                  [0.08, 10.22, 9.56, 6.26, 18.92, 11.24],
    "RDT":                  [0.06, 7.46, 8.08, 2.52, 10.90, 8.42],
    "X-VLA":                [0.04, 5.85, 3.37, 1.34, 10.55, 7.18],
}

# Table 4: success rate per reasoning category
# Columns as printed in the paper: Geometric, Physical, Memory.
TABLE4_BY_CATEGORY = {  # [Geometric, Physical, Memory]
    "DM0":                  [0.47, 0.35, 0.13],
    "GigaBrain0.1":         [0.29, 0.47, 0.06],
    "LingBot-VLA w/ depth": [0.41, 0.12, 0.19],
    "Pi0.5":                [0.24, 0.18, 0.00],
    "GR00T-N1.6":           [0.18, 0.06, 0.13],
    "Pi0":                  [0.24, 0.00, 0.00],
    "RDT":                  [0.06, 0.06, 0.06],
    "X-VLA":                [0.12, 0.00, 0.00],
}
CATEGORY_NAMES = ["Geometric", "Physical", "Memory"]

# Table 5: Spearman's rho between the 5 metrics (reasoning tasks)
# Transcribed from the paper's upper-triangular print, mirrored into the full
# symmetric matrix:
#                Comp  Space  Time  Smooth  Safety
#   Comp         1.00  0.90   0.86  0.90    0.95
#   Space              1.00   0.93  0.98    0.93
#   Time                      1.00  0.93    0.88
#   Smooth                          1.00    0.98
#   Safety                                  1.00
TABLE5_METRIC_NAMES = ["Comp", "Space", "Time", "Smooth", "Safety"]
TABLE5_SPEARMAN = [
    [1.00, 0.90, 0.86, 0.90, 0.95],
    [0.90, 1.00, 0.93, 0.98, 0.93],
    [0.86, 0.93, 1.00, 0.93, 0.88],
    [0.90, 0.98, 0.93, 1.00, 0.98],
    [0.95, 0.93, 0.88, 0.98, 1.00],
]

# --- §4.4 Human-Automation Alignment --------------------------------------
# The reported headline for this study: the mean Spearman rho between the
# three-expert human consensus and the automated metrics, across 8 models and
# the 5 dimensions. Per-dimension rho and the inter-rater Kendall's W are
# produced by the study harness (robogym.analysis.human_study).
MEAN_SPEARMAN = 0.90      # §4.4: "averages rho = 0.9"
HUMAN_STUDY_N_EXPERTS = 3
HUMAN_STUDY_N_MODELS = 8
HUMAN_STUDY_N_DIMENSIONS = 5

ALL_MODELS = list(TABLE2_AUGMENTED_LIBERO.keys())
