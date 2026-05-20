"""Reported numbers from the paper, used for analysis / figure regeneration.

Tables 2-5 hold the values printed in the main text. Table 6 and the
inter-rater Kendall's W are from Appendix B. ``REASONING_CATEGORY_SPLIT``
records how the 50 System-2 tasks are distributed across the three
reasoning categories, so per-category success counts in Table 4 are
interpretable.
"""

METRICS = ["SR", "Comp", "Space", "Time", "Smooth", "Safety"]

# Table 2: performance on Augmented LIBERO.
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

# Table 3: performance on the 50-task System-2 reasoning suite.
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

# Table 4: success rate per reasoning category (Geometric, Physical, Memory).
TABLE4_BY_CATEGORY = {
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

# How the 50 System-2 tasks split across the three reasoning categories.
REASONING_CATEGORY_SPLIT = {"geometric": 18, "physical": 18, "memory": 14}
REASONING_FAMILY_COUNTS = {
    "tangram_assembly": 9, "number_block": 9,
    "maze_navigation": 9, "seesaw_weight": 9,
    "color_hanoi": 7, "sequential_counting": 7,
}

# Table 5: Spearman rho between the five core metrics on the reasoning suite.
TABLE5_METRIC_NAMES = ["Comp", "Space", "Time", "Smooth", "Safety"]
TABLE5_SPEARMAN = [
    [1.00, 0.90, 0.86, 0.90, 0.95],
    [0.90, 1.00, 0.93, 0.98, 0.93],
    [0.86, 0.93, 1.00, 0.93, 0.88],
    [0.90, 0.98, 0.93, 1.00, 0.98],
    [0.95, 0.93, 0.88, 0.98, 1.00],
]

# Appendix B Table 6: per-dimension Spearman rho between the three-expert
# human consensus and the automated metrics; the mean across dimensions is
# the headline 0.90 in Section 4.4.
TABLE6_HUMAN_AUTOMATION = {
    "Comp": 0.88, "Space": 0.89, "Time": 0.92,
    "Smooth": 0.86, "Safety": 0.95,
}
MEAN_SPEARMAN = 0.90
INTER_RATER_KENDALLS_W = 0.89

HUMAN_STUDY_N_EXPERTS = 3
HUMAN_STUDY_N_MODELS = 8
HUMAN_STUDY_N_DIMENSIONS = 5

ALL_MODELS = list(TABLE2_AUGMENTED_LIBERO.keys())
