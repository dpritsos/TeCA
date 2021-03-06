
import sys
import numpy as np
sys.path.append('../../teca')

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

clusters_y = [1]*500 + [2]*500 + [3]*500 + [4]*200 + [5]*200 + [6]*200 + [7]*200
clusters_y = np.array(clusters_y)
categories_y = np.random.permutation(clusters_y)
categories_y = np.array(categories_y)
# categories_y = clusters_y

print categories_y

print bcubed_pr_scores(clusters_y, categories_y)
