
import sys
import numpy as np

sys.path.append('../../teca')

print sys.path

from analytics.metrix import bcubed_pr_scores

clusters_y = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    # 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    # 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    # 5, 5, 5, 5, 5, 5, 5, 5, 5, 5
])
categories_y = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    # 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    # 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    # 11, 11, 11, 11, 11, 11, 11, 11, 11, 11
])

print bcubed_pr_scores(clusters_y, categories_y)
