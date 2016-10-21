#!/usr/bin/env python

import sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
import collections as coll
import sklearn.metrics as skm
# from data_retrieval.ocsvmedata import get_predictions as get_ocsvme
# from sklearn import grid_search
# from sklearn.metrics import roc_curve as roc
# from sklearn.metrics import precision_recall_curve

sys.path.append('../../teca')
sys.path.append('../../../DoGSWrapper/dogswrapper')

from analytics.metrix import bcubed_pr_scores
from data_retrieval.semisupervised_data import get_predictions
import base.param_combs as param_comb

params_range = coll.OrderedDict([
   #  ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
   ('train_split_step_method', [
      # [0.3, 0.1, 'rndred_trn_rest4_test'],
      [0.10, 0.10, 'rndred_trn_fixed_test'],
   ]),
   ('vocab_size', [50000]),  # 10, 50, 500, 5000, 10000,
   ('max_iter', [100]),  # 30, 100, 300
   ('converg_diff', [0.001]),  # 0.0005, 0.005, 0.01, 0.05, 0.1, 0.5
   ('learing_rate', [0.003]),  # 0.3, ,0.03, , 0.001
   ('#', [0])
])

h5df_str = str(
    '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/SemiSupClust_3W_7Genres_text/' +
    'HMRFKmeans_3W_7Genres_10%_50k_text_cy.h5'
    # '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/
    # '/home/dimitrios/Synergy-Crawler/SANTINIS/
    # '/home/dimitrios/Synergy-Crawler/SANTINIS/
    # '/home/dimitrios/Synergy-Crawler/KI-04/
    # '/home/dimitrios/Synergy-Crawler/KI-04/
)

h5df = tb.open_file(h5df_str, 'r')


symbol = ['o', 'v', '^', '+', 'x', 's', '*', '<', '>', 'H',
          '1', '2', '3', '4', 'D', 'h', '8', 'd', 'p', '.', ',']

line_type = ['--', '--', '--', '--', '-', '-', '-', '-', '-', '--',
             '--', '--', '--.', '-', '-', '-', '-', '--', '--', '--', '--']

color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k',
         'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']

color2 = ['r', 'g', 'b', 'k', 'c', 'y', 'm', 'r', 'g', 'b', 'k', 'c', 'y', 'm']

# fg1 = plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
# ax1 = fg1.add_subplot(111)

# fg2 = plt.figure(num=2, figsize=(30, 8), dpi=80, facecolor='w', edgecolor='k')
# ax2 = fg2.add_subplot(111)

# i = 0
# bar_width = 0.15

for params_lst, params_path in \
    zip(param_comb.ParamGridIter(params_range, 'list'),
        param_comb.ParamGridIter(params_range, 'path')):

        clstr_y, clss_y, clstr_params = get_predictions(h5df, params_path, class_tag=None)

        clstr_y = clstr_y.reshape(1, clstr_y.shape[0])

        pre_bc, rec_bc, size_per_clstr, size_per_cats = bcubed_pr_scores(clstr_y[0], clss_y[0])

        print str(clstr_y.shape[1]) + ", " + ", ".join([str(i) for i in size_per_clstr[1::]]) + ", " + ", ".join([str(i) for i in size_per_cats[1::]]) + ", " + ", ".join([str(i) for i in clstr_params]) + ", " +  str(params_lst[1]) + ", " + str(pre_bc) + ", " + str(rec_bc)

        # ", ".join([str(i) for i in params_lst])

        # plt.locator_params(nbins=4)
        # ax1.plot(
        #     x, y,
        #     color[i] + line_type[i] + symbol[i], linewidth=1,
        #     markeredgewidth=1,
        #     # label="KI04 - 3Words"
        #     # "(" + str(i) + ") Feat " + str(params_lst[2]) + \
        #     # " - " + str(params_lst[3])
        # )

        # ax1.title("SANTINIS")
        # ax1.yticks([ .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.00])
        # .10, .15, .20, .25, .30, .35, .40, .45,
        # ax1.grid(True)
        # ax1.legend(loc=4, fancybox=True, shadow=True)
        # ax1.legend(loc=3, bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)

        # hist, bins = np.histogram(pred_y, 12)
        # ax2.bar(bins[0:12] + bar_width , hist, width=0.1, color=color2[i])
        # i, zdir='y', alpha=0.8, align='center', width=bins[1]-bins[0])

"""
plt.xticks(
    bins[0:12] + bar_width,
    (
        'Dont Know', "blog", "eshop", "faq", "frontpage", "listing", "php", "spage",
        "diy_mini", "editorial", "feat_articles", "short_bio"
    )
)
"""
# plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.02])
# plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.02])


# plt.show()

h5df.close()
