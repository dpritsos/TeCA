

# !/usr/bin/env python

import tables as tb
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../teca')
sys.path.append('../../../DoGSWrapper/dogswrapper')

from analytics.metrix import pr_curve, pr_curve_macro, reclev11_max
from data_retrieval.data import multiclass_res
from data_retrieval.data import rfse_multiclass_multimeasure_res
from data_retrieval.data import rfse_onevsall_res
from data_retrieval.data import rfse_onevsall_multimeasure_res

# Funciton for creating the parameter paths requred for get_predicitons()
def plist2ppath(params_lst, ensbl='RFSE'):

    if ensbl == 'RFSE':

        pnames = ['vocab_size', 'features_size', 'Sigma', 'Iterations', 'KFold']

        return ''.join(['/' + n + str(v).replace('.', '') for n, v in zip(pnames, params_lst)])

    elif ensbl == 'OCSVME':

        pnames = ['vocab_size', 'features_size', 'nu', 'KFold']

        return ''.join(['/' + n + str(v).replace('.', '') for n, v in zip(pnames, params_lst)])

    else:
        raise Exception('Invalid Ensebmle Name')


kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 7Genres
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/MacroPRC11AVG_RFSE_OCSVME_SANTINIS.eps'
# fig_save_file = '/home/dimitrios/MacroPRC11AVG_RFSE_OCSVME_7Genres.eps'

comb_lst = [
    # ['RFSE', '3Words', 'SANTINIS', 'MinMax', [50000, 5000, 0.7, 100, '']],
    ['RFSE', '1Words', 'SANTINIS', 'Cosine', [50000, 5000, 0.5, 100, '']],
    ['RFSE', '1Words', 'SANTINIS', 'Cosine', [5000, 1000, 0.5, 100, '']],
    ['OCSVME', '1Words', 'SANTINIS', '', [50000, 5000, 0.1, '']],
    ['OCSVME', '1Words', 'SANTINIS', '', [50000, 500, 0.1, '']],


]

plt_dsp_attr = [
    ['black', '-', 'o', "W1G - RFSE - F1"],
    ['orange', '-', 'o', "W1G - RFSE - AUC"],
    ['purple', '--', 'o', "W1G - OCSVM - F1"],
    ['lime', '--', 'o', "W1G - OCSVM - AUC"],
]

# # # #  The Ploting Process Starts Here # # # #
fig = plt.figure(num=1, figsize=(12, 8), facecolor='w', edgecolor='k')  # dpi=300,
ax = fig.add_subplot(111)

annots_lst = list()
labels_lst = list()

for i, comb_val in enumerate(comb_lst):

    # Selecting filepath
    if comb_val[0] == 'RFSE':

        if comb_val[2] == '7Genres':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_'

        elif comb_val[2] == 'KI04':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/KI-04/RFSE_'

        elif comb_val[2] == 'SANTINIS':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_'

    elif comb_val[0] == 'OCSVME':

        if comb_val[2] == '7Genres':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_'

        elif comb_val[2] == 'KI04':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_'

        elif comb_val[2] == 'SANTINIS':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_'

    h5d_fl = h5d_fl + comb_val[1] + '_' + comb_val[2]

    #  Selecting files to open and setting the mix flag on/off
    if comb_val[3] == 'Comb':
        h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
        h5d_fl2 = tb.open_file(h5d_fl + '_minmax.h5', 'r')

    elif comb_val[3] == 'MinMax':
        h5d_fl1 = tb.open_file(h5d_fl + '_minmax.h5', 'r')

    elif comb_val[3] == 'Cosine' or comb_val[3] == '':
        h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')

    else:
        raise Exception("Option: " + comb_val[3] + " is not valid for Measure Option")

    # Getting the predictions
    if comb_val[3] == 'Comb':

        # Building the parapmeters path
        params_path = plist2ppath(comb_val[4], ensbl=comb_val[0])

        pred_scores, expd_y, pred_y = rfse_multiclass_multimeasure_res(
            # h5d_fl1, h5d_fl2, kfolds, params_path, comb_val[4][2],
            # genre_tag=None, binary=True, strata=None
            h5d_fl1, h5d_fl2, kfolds, params_path, binary=False, strata=None
            #  binary=False <- for Micro
        )

    else:

        # Building the parapmeters path
        params_path = plist2ppath(comb_val[4], ensbl=comb_val[0])

        pred_scores, expd_y, pred_y = multiclass_res(
            h5d_fl1, kfolds, params_path, binary=False, strata=None
        )

    # Closing the h5d files.
    if comb_val[3] == 'Comb':
        h5d_fl1.close()
        h5d_fl2.close()
    else:
        h5d_fl1.close()

    # Creating the Actual PRC.
    # y, x, t = pr_curve(expd_y, pred_scores, full_curve=True, is_truth_tbl=True)

    # Creating the Actual MACRO PRC.
    y, x, t = pr_curve_macro(
        expd_y, pred_y, pred_scores, full_curve=True
    )

    # Getting the max 11 Recall Leves in TREC way.
    # if i == 0:
    y, x = reclev11_max(y, x, trec=False)

    # Selecting array indices with non-zero cells.
    non_zero_idx = np.where(y > 0)

    # # # Do the Plotting
    linestyle = {
        "color": plt_dsp_attr[i][0],
        "linestyle": plt_dsp_attr[i][1],
        "marker": plt_dsp_attr[i][2],
        "linewidth": 2,
        "markeredgewidth": 2,
        'markeredgecolor': 'white',
    }

    ax.plot(x[non_zero_idx], y[non_zero_idx], **linestyle)

    annots_lst.append(mlines.Line2D([], [], markersize=0, linewidth=3, color=plt_dsp_attr[i][0]))
    labels_lst.append(plt_dsp_attr[i][3])

    # lndump = mlines.Line2D([], [], markersize=0, linewidth=0)

    """
    ax.annotate(
    'F1=0.782, AUC=0.843',
    xy=(0.12, 0.98), xytext=(0.2, 0.85), fontsize=16,
    arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
    bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
    )
    """

# Give the poper attributes for better ploting
ax.yaxis.grid()

lndump = mlines.Line2D([], [], markersize=0, linewidth=0)
annots_lst.append(lndump)
annots_lst.append(lndump)
labels_lst.append("")
labels_lst.append("")

plt.legend(
    annots_lst,
    labels_lst,
    bbox_to_anchor=(0.0, 1.01, 1.0, 0.101),
    loc=3, ncol=3, mode="expand", borderaxespad=0.0,
    fancybox=False, shadow=False, fontsize=14
).get_frame().set_linewidth(0.0)

plt.yticks(fontsize=12)
plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
plt.ylabel('Precision', fontsize=14)
plt.xlabel('Recall', fontsize=14)
# plt.tight_layout()

# Saving the ploting to File
# plt.savefig(fig_save_file, bbox_inches='tight')

plt.show()
