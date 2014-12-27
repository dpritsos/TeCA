#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import matplotlib.pyplot as plt
import numpy as np

from analytics.metrix import pr_curve, reclev11_max
from data_retrieval.rfsedata import get_predictions
from data_retrieval.rfsemixdata import get_predictions as get_predictions_mix


#Symbols for plosting.
symbol = ['o', 'v', '^', '*', '<', 's', '+', 'x', '>', 'H', '1', '2', '3', '4', 'D', 'h',
          '8', 'd', 'p', '.', ',']
line_type = ['--', '--', '--', '-', '-', '-', '--', '--', '--', '-', '-', '-', '--.', '-',
             '-', '-', '-', '--', '--', '--', '--']


#Funciton for creating the parameter paths requred for get_predicitons()
def plist2ppath(params_lst, ensbl='RFSE'):

    if ensbl == 'RFSE':

        pnames = ['vocab_size', 'features_size', 'Sigma', 'Iterations']

        return ''.join(['/' + n + str(v).replace('.', '') for n, v in zip(pnames, params_lst)])

    elif ensbl == 'OCSVME':

        pnames = ['vocab_size', 'features_size', 'nu']

        return ''.join(['/' + n + str(v).replace('.', '') for n, v in zip(pnames, params_lst)])

    else:
        raise Exception('Invalid Ensebmle Name')


kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


#### Paramter Combination for Cases to be ploted


comb_lst = [
    ['RFSE', '3Words', '7Genres', '', [100000, 50000, 0.5, 50]],
]


####

#### The Ploting Process Starts Here ####
fig = plt.figure(num=1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)

for i, comb_val in enumerate(comb_lst):

    #Selecting filepath
    if comb_val[0] == 'RFSE':

        if comb_val[2] == '7Genres':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_'

        elif comb_val[2] == 'KI04':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/KI-04/RFSE_'

        else:
            h5d_fl = '/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_'

    elif comb_val[0] == 'OCSVME':

        if comb_val[2] == '7Genres':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_'

        elif comb_val[2] == 'KI04':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_'

        else:
            h5d_fl = '/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_'

    h5d_fl = h5d_fl + comb_val[1] + '_' + comb_val[2]

    # Selecting files to open and setting the mix flag on/off
    mix = False

    if comb_val[3] == 'MIX':
        h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
        h5d_fl2 = tb.open_file(h5d_fl + '_minmax.h5', 'r')
        mix = True

    elif comb_val[3] == 'MinMax':
        h5d_fl1 = tb.open_file(h5d_fl + '_minmax.h5', 'r')
        h5d_fl2 = h5d_fl1

    else:
        h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
        h5d_fl2 = h5d_fl1

    #Getting the predictions
    if comb_val[3] == 'MIX':

        #Building the parapmeters path
        params_path = plist2ppath(comb_val[4], ensbl=comb_val[0])

        pred_scores, expd_y, pred_y = get_predictions_mix(
            h5d_fl1, h5d_fl2, kfolds, params_path, comb_val[4][2],
            genre_tag=None, binary=True, strata=None
        )

    else :

        #Building the parapmeters path
        params_path = plist2ppath(comb_val[4], ensbl=comb_val[0])

        pred_scores, expd_y, pred_y = get_predictions(
            h5d_fl1, kfolds, params_path, genre_tag=None, binary=True, strata=None
        )

    #Closing the h5d files.
    if comb_val[3] == 'MIX':
        h5d_fl1.close()
        h5d_fl2.close()
    else:
        h5d_fl1.close()

    ###Create the Actual PRC.
    y, x, t = pr_curve(expd_y, pred_scores, full_curve=True, is_truth_tbl=True)

    ###Get the max 11 Recall Leves in TREC way.
    y, x = reclev11_max(y, x)

    ###Do the Plotting
    ax.plot(
        x, y,
        'k' + line_type[i] + symbol[i], linewidth=1,
        markersize=10,
        label = comb_val[1] +'-'+ comb_val[3].replace('', 'Cosine').replace('MIX', 'Comb') +'-'+ 'F1'
    )

    """
    ax.annotate(
    'F1=0.782, AUC=0.843',
    xy=(0.12, 0.98), xytext=(0.2, 0.85), fontsize=16,
    arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
    bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
    )

    ax.annotate(
        'F1=0.770, AUC=0.829',
        xy=(0.08, 0.925), xytext=(0.05, 0.75), fontsize=16,
        arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
        bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
    )

    ax.annotate(
        'F1=0.775, AUC=0.845',
        xy=(0.83, 0.95), xytext=(0.65, 0.85), fontsize=16,
        arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
        bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
    )

    ax.annotate(
        'F1=0.601, AUC=0.560',
        xy=(0.8, 0.743), xytext=(0.5, 0.65), fontsize=16,
        arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
        bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
    )
    """


#Give the poper attributes for better ploting
plt.grid(True)
#plt.legend(loc='upper left', bbox_to_anchor=(0.62, 0.4), ncol=1, fancybox=True, shadow=True, fontsize=16)
plt.legend(loc=4, fancybox=True, shadow=True, fontsize=16)
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], fontsize=16)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=16)
plt.tight_layout()

#Saving the ploting to File
#plt.savefig(fig_save_file, bbox_inches='tight')

plt.show()
