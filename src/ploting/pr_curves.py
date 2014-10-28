#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import matplotlib.pyplot as plt

from analytics.metrix import pr_curve, reclev_averaging
    #roc_curve, reclev_max_around,
    #reclev_nearest, smooth_linear
#from data_retrieval.rfsedata import get_predictions
from data_retrieval.rfsemixdata import get_predictions
import base.param_combs as param_comb
import collections as coll
import numpy as np
#from data_retrieval.ocsvmedata import get_predictions as get_ocsvme
#from sklearn import grid_search
#from sklearn.metrics import roc_curve as roc
#from sklearn.metrics import precision_recall_curve

kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_od = coll.OrderedDict([
    ('vocab_size', [100000]),  # [5000, 10000, 50000, 100000]),
    ('features_size', [1000, 5000, 10000, 50000, 90000]),  # 1000, 5000, 10000, 50000, 90000]
    #'3.Bagging', [0.66],
    # ('nu', [0.5]) #, 0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9])
    ('Sigma', [0.7]),  # [0.5, 0.7, 0.9])
    ('Iterations', [100])  # [10, 50, 100]
])

h5d_fl = str(
    #/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_3Words_7Genres'
    #'/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_3Words_7Genres'
    '/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_3Words_SANTINIS'
    #'/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_3Words_SANTINIS'
    #'/home/dimitrios/Synergy-Crawler/KI-04/RFSE_1Words_KI04_minmax'
    #'/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_3Words_KI04'
)

h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
h5d_fl2 = tb.open_file(h5d_fl + '_minmax.h5', 'r')


symbol = ['o', 'v', '^', '+', 'x', 's', '*', '<', '>', 'H',
          '1', '2', '3', '4', 'D', 'h', '8', 'd', 'p', '.', ',']

line_type = ['--', '--', '--', '--', '-', '-', '-', '-', '-', '--',
             '--', '--', '--.', '-', '-', '-', '-', '--', '--', '--', '--']

color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k',
         'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']

color2 = ['r', 'g', 'b', 'k', 'c', 'y', 'm', 'r', 'g', 'b', 'k', 'c', 'y', 'm']

fg1 = plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
ax1 = fg1.add_subplot(111)
#fg2 = plt.figure(num=2, figsize=(30, 8), dpi=80, facecolor='w', edgecolor='k')
#ax2 = fg2.add_subplot(111)

i = 0
bar_width = 0.15

for params_lst, params_path in \
    zip(param_comb.ParamGridIter(params_od, 'list'),
        param_comb.ParamGridIter(params_od, 'path')):

    if params_lst[0] > params_lst[1]:

        #pred_scores, expd_y, pred_y = get_predictions(
        #    h5d_fl1, kfolds, params_path, genre_tag=None, binary=False
        #)

        pred_scores, expd_y, pred_y = get_predictions(
            h5d_fl1, h5d_fl2, kfolds, params_path, params_lst[2], gnr_num=12,
            genre_tag=None, binary=True
        )

        #pred_scores, expd_y, pred_y = get_ocsvme(res_h5file, kfolds, params_path, genre_tag=None)

        #y, x, t = roc_curve(expd_y, pred_scores, full_curve=False)
        y, x, t = pr_curve(expd_y, pred_scores, full_curve=True)
        # y, x, t = precision_recall_curve(expd_y, pred_scores) #<== They are
        # not 100% percent the same.

        # Smoothing out the Precision (y axis) of the P-R Curve.
        # CRITICAL: The P sould be inverted from the lowest to
        # the highest values*.
        # *( it suppose the higest values to be normally fist in order )
        #y = smooth_linear(y[::-1])

        # Inverting the y (i.e. Precition) axis values after has been smoothed out.
        #y = y[::-1]

        # OR

        #y, x = smooth_linear(y[::-1], x[::-1])
        #y, x = y[::-1], x[::-1]

        y, x = reclev_averaging(y, x)

        #y, x = reclev_nearest(y, x)

        #y, x = reclev_max_around(y, x)

        # plt.locator_params(nbins=4)
        ax1.plot(
            x, y,
            color[i] + line_type[i] + symbol[i], linewidth=1,
            markeredgewidth=1, label="(" + str(i) + ") Feat " + str(params_lst[1])
        )
        # ax1.title("SANTINIS")
        # ax1.yticks([ .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.00])
        # .10, .15, .20, .25, .30, .35, .40, .45,
        ax1.grid(True)
        ax1.legend(loc=4, fancybox=True, shadow=True)
        #ax1.legend(loc=3, bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)

        hist, bins = np.histogram(pred_y, 12)

        # print len(pred_y[pred_y == 0]), hist, bins

        #tmp = np.zeros(len(bins+1))
        #tmp[0:-1] = hist[::]
        #hist = tmp

        # ax2.bar(bins[0:12] + bar_width , hist, width=0.1, color=color2[i])
        # i, zdir='y', alpha=0.8, align='center', width=bins[1]-bins[0])
        i += 1
        bar_width += 0.1

"""
plt.xticks(
    bins[0:12] + bar_width,
    (
        'Dont Know', "blog", "eshop", "faq", "frontpage", "listing", "php", "spage",
        "diy_mini", "editorial", "feat_articles", "short_bio"
    )
)
"""

#print y, x


plt.show()

h5d_fl1.close()
h5d_fl2.close()
