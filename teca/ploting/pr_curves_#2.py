

# !/usr/bin/env python

import tables as tb
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


# Symbols for plosting.
symbol = ['o', '*', '+', '^', '<', 's', '+', 'x', '>', 'H', '1', '2', '3', '4', 'D', 'h',
          '8', 'd', 'p', '.', ',']
line_type = ['-', '--', '--', '-', '-', '-', '--', '--', '--', '-', '-', '-', '--.', '-',
             '-', '-', '-', '--', '--', '--', '--']




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



### Parameter Combination for Cases to be plotted

# ++++++++++++++++++++++++++++++
#   RFSE
# ++++++++++++++++++++++++++++++


"""
# 7Genres 3W for every Distance Measure.
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_7Genres_3W_F1Based_11AVG.eps'

comb_lst = [
    ['RFSE', '3Words', '7Genres', 'Cosine', [100000, 50000, 0.5, 50]],
    ['RFSE', '3Words', '7Genres', 'MinMax', [100000, 90000, 0.5, 10]],
    ['RFSE', '3Words', '7Genres', 'Comb', [100000, 50000, 0.5, 100]],
    ['OCSVME', '3Words', '7Genres', '', [10000, 5000, 0.1]]
]

plt_dsp_attr = [
    ['k' + line_type[1] + symbol[0], 2, 14, "3W - Cosine"],
    ['k' + line_type[1] + symbol[1], 2, 14, "3W - MinMax"],
    ['k' + line_type[1] + symbol[3], 2, 14, "3W - Comb"],
    ['k' + line_type[5], 3, 0, "3W - Baseline"]
]

leg_pos = 3


# 7Genres for every Similarity Measure & Document Representation with KI04 Optimal Parameters.
fig_save_file =
#'/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_7Genres_ParamsMostOccured_F1Based_11AVG.eps'

comb_lst = [
    ['RFSE', '4Chars', 'SANTINIS', 'Cosine', [50000, 1000, 0.7, 100]],
    ['RFSE', '4Chars', 'SANTINIS', 'Cosine', [50000, 1000, 0.9, 100]],
    # ['RFSE', '1Words', '7Genres', 'MinMax', [100000, 5000, 0.5, 100]],
    # ['RFSE', '4Chars', '7Genres', 'MinMax', [100000, 5000, 0.5, 100]],
    # ['RFSE', '1Words', '7Genres', 'Comb', [100000, 5000, 0.5, 100]],
    # ['RFSE', '4Chars', '7Genres', 'Comb', [100000, 5000, 0.5, 100]],
    # ['OCSVME', '4Chars', '7Genres', '', [100000, 50000, 0.1]]
]


plt_dsp_attr = [
    ['k' + line_type[1] + symbol[0], 2, 14, "1W - Cosine"],
    ['k' + line_type[1] + symbol[1], 2, 14, "4C - Cosine"],
    ['k' + line_type[1] + symbol[3], 2, 14, "1W - MinMax"],
    ['k' + line_type[1] + symbol[4], 2, 14, "4C - MinMax"],
    #['m' + line_type[1] + symbol[5], 2, 14, "1W - Comb"],
    #['y' + line_type[1] + symbol[6], 2, 14, "4C - Comb"],
    ['k' + line_type[5], 3, 0, "4C - Baseline"]
]

leg_pos = 4


# KI04 1W for every Distance Measure.
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_KI04_1W_F1Based_11AVG.eps'

comb_lst = [
    ['RFSE', '1Words', 'KI04', 'Cosine', [100000, 50000, 0.5, 50]],
    ['RFSE', '1Words', 'KI04', 'MinMax', [10000, 5000, 0.5, 100]],
    ['RFSE', '1Words', 'KI04', 'Comb', [10000, 5000, 0.5, 100]],
    ['OCSVME', '1Words', 'KI04', '', [10000, 1000, 0.1]],
]

plt_dsp_attr = [
    ['k' + line_type[1] + symbol[0], 2, 14, "1W - Cosine"],
    ['k' + line_type[1] + symbol[1], 2, 14, "1W - MinMax"],
    ['k' + line_type[1] + symbol[3], 2, 14, "1W - Comb"],
    ['k' + line_type[5], 3, 0, "1W - Baseline"]
]

leg_pos = 1



# SANTINIS 3W for every Distance Measure.
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_SANTINIS_3W_F1Based_11AVG.eps'

comb_lst = [
    ['RFSE', '3Words', 'SANTINIS', 'Cosine', [50000, 10000, 0.7, 100]],
    ['RFSE', '3Words', 'SANTINIS', 'MinMax', [50000, 5000, 0.7, 100]],
    ['RFSE', '3Words', 'SANTINIS', 'Comb', [100000, 1000, 0.5, 100]],
    ['OCSVME', '3Words', 'SANTINIS', '', [50000, 10000, 0.1]]
]

plt_dsp_attr = [
    ['k' + line_type[1] + symbol[0], 2, 14, "3W - Cosine"],
    ['k' + line_type[1] + symbol[1], 2, 14, "3W - MinMax"],
    ['k' + line_type[1] + symbol[3], 2, 14, "3W - Comb"],
    ['k' + line_type[5], 3, 0, "3W - Baseline"]
]

leg_pos = 3
"""
"""
# SANTINIS for every Similarity Measure & Document Representation with KI04 Optimal Parameters.
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_SANTINIS_ParamsMostOccured_F1Based_11AVG.eps'

comb_lst = [
    ['RFSE', '1Words', 'SANTINIS', 'Cosine', [100000, 5000, 0.5, 100]],
    ['RFSE', '4Chars', 'SANTINIS', 'Cosine', [100000, 5000, 0.5, 100]],
    ['RFSE', '1Words', 'SANTINIS', 'MinMax', [100000, 5000, 0.5, 100]],
    ['RFSE', '4Chars', 'SANTINIS', 'MinMax', [100000, 5000, 0.5, 100]],
    #['RFSE', '1Words', 'SANTINIS', 'Comb', [100000, 5000, 0.5, 100]],
    #['RFSE', '4Chars', 'SANTINIS', 'Comb', [100000, 5000, 0.5, 100]],
    ['OCSVME', '4Chars', 'SANTINIS', '', [10000, 5000, 0.5]]
]


plt_dsp_attr = [
    ['k' + line_type[1] + symbol[0], 2, 14, "1W - Cosine"],
    ['k' + line_type[1] + symbol[1], 2, 14, "4C - Cosine"],
    ['k' + line_type[1] + symbol[3], 2, 14, "1W - MinMax"],
    ['k' + line_type[1] + symbol[4], 2, 14, "4C - MinMax"],
    #['m' + line_type[1] + symbol[5], 2, 14, "1W - Comb"],
    #['y' + line_type[1] + symbol[6], 2, 14, "4C - Comb"],
    ['k' + line_type[5], 3, 0, "4C - Baseline"]
]

leg_pos = 4
"""
"""
# 7Genre Distance Measure Cosine
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_7Genres_Cosine_F1Based_11AVG.eps'

comb_lst = [
    ['RFSE', '3Words', '7Genres', 'Cosine', [100000, 50000, 0.5, 50]],
    ['RFSE', '1Words', '7Genres', 'Cosine', [100000, 50000, 0.5, 100]],
    ['RFSE', '4Chars', '7Genres', 'Cosine', [50000, 5000, 0.5, 50]]
]

plt_dsp_attr = [
    ['k' + line_type[1] + symbol[0], 2, 14, "3W - Cosine"],
    ['k' + line_type[1] + symbol[1], 2, 14, "1W - Cosine"],
    ['k' + line_type[1] + symbol[3], 2, 14, "4C - Cosine"],
]

leg_pos = 3



# KI04 Distance Measure MinMax
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_KI04_MinMax_F1Based_11AVG.eps'

comb_lst = [
    ['RFSE', '3Words', 'KI04', 'MinMax', [100000, 90000, 0.5, 50]],
    ['RFSE', '1Words', 'KI04', 'MinMax', [10000, 5000, 0.5, 100]],
    ['RFSE', '4Chars', 'KI04', 'MinMax', [10000, 1000, 0.5, 100]]
]

plt_dsp_attr = [
    ['k' + line_type[1] + symbol[0], 2, 14, "3W - MinMax"],
    ['k' + line_type[1] + symbol[1], 2, 14, "1W - MinMax"],
    ['k' + line_type[1] + symbol[3], 2, 14, "4C - MinMax"],
]

leg_pos = 3



# SANTINIS Distance Measure MinMax
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_SANTINIS_MinMax_F1Based_11AVG.eps'

comb_lst = [
    ['RFSE', '3Words', 'SANTINIS', 'MinMax', [50000, 5000, 0.7, 100]],
    ['RFSE', '1Words', 'SANTINIS', 'MinMax', [100000, 10000, 0.7, 100]],
    ['RFSE', '4Chars', 'SANTINIS', 'MinMax', [100000, 5000, 0.9, 100]]
]

plt_dsp_attr = [
    ['k' + line_type[1] + symbol[0], 2, 14, "3W - MinMax"],
    ['k' + line_type[1] + symbol[1], 2, 14, "1W - MinMax"],
    ['k' + line_type[1] + symbol[3], 2, 14, "4C - MinMax"],
]

leg_pos = 3



# 7Genre F1 VS F05
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PRC_7Genres_F1vsF05_11AVG.eps'

comb_lst = [
    ['RFSE', '3Words', '7Genres', 'Cosine', [100000, 50000, 0.5, 50]],
    ['RFSE', '3Words', '7Genres', 'Cosine', [100000, 50000, 0.7, 50]],
    ['RFSE', '3Words', '7Genres', 'Cosine', [100000, 50000, 0.5, 100]],
    ['OCSVME', '4Chars', '7Genres', '', [100000, 50000, 0.1]],
    ['OCSVME', '3Words', '7Genres', '', [50000, 10000, 0.1]]
]

plt_dsp_attr = [
    ['k' + line_type[1] + symbol[0], 2, 14, "3W - Cos - F1"],
    ['k' + line_type[1] + symbol[1], 2, 14, "3W - Cos - F0.5"],
    ['k' + line_type[1] + symbol[3], 2, 14, "3W - Cos - AUC"],
    ['k' + line_type[3] + symbol[5], 2, 14, "4C - OCSVME - F1"],
    ['k' + line_type[3] + symbol[4], 2, 14, "3W - OCSVME - F0.5"]
]

leg_pos = 3





# KI04 F1 VS F05
# ++++++++++++++++++++ NOT PLOTED FOR NOW +++++++++++++++++++++++++++++++++++
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PRC_KI04_F1vsF05_11AVG.eps'

plt_dsp_attr = [
    ['k' + line_type[0] + symbol[0], 2, 14, "1W - MinMax - F1"],
    ['k' + line_type[1] + symbol[1], 2, 14, "1W - Cos - F0.5"],
    ['k' + line_type[2] + symbol[2], 2, 14, "4C - Cos - F0.5"],
    ['k' + line_type[3] + symbol[3], 2, 14, "1W - OCSVME - F1"],
    ['k' + line_type[4] + symbol[4], 2, 14, "1W - OCSVME - F0.5"],
    ['k' + line_type[5] + symbol[5], 2, 14, "3W - OCSVME - P"]
]



# SANTINIS F1 VS F05
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PRC_SANTINIS_F1vsF05_11AVG.eps'

comb_lst = [


    ['RFSE', '3Words', 'SANTINIS', 'MinMax', [50000, 5000, 0.7, 100]],
    # ['RFSE', '3Words', 'SANTINIS', 'MinMax', [50000, 5000, 0.7, 100]],


    ['RFSE', '3Words', 'SANTINIS', 'Cosine', [100000, 1000, 0.5, 50]],
    # ['RFSE', '1Words', 'SANTINIS', 'Cosine', [50000, 1000, 0.5, 50]],
    ['OCSVME', '3Words', 'SANTINIS', '', [50000, 10000, 0.1]],
    ['OCSVME', '3Words', 'SANTINIS', '', [100000, 90000, 0.9]]
]

plt_dsp_attr = [


    ['k' + line_type[3] + symbol[6], 2, 14, "3W - MinMax - F1"],
    # ['k' + line_type[1] + symbol[0], 2, 14, "3W - MinMax - F0.5"],


    ['k' + line_type[1] + symbol[2], 2, 14, "3W - Cosine - AUC"],
    # ['k' + line_type[1] + symbol[3], 2, 14, "1W - Cosine - mP"],
    ['k' + line_type[3] + symbol[5], 2, 14, "3W - OCSVME - F1"],
    ['k' + line_type[1] + symbol[4], 2, 14, "3W - OCSVME - mP"]
]

leg_pos = 3



# ++++++++++++++++++++++++++++++
#   OCSVME
# ++++++++++++++++++++++++++++++

"""
# 7Genres
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/MacroPRC11AVG_RFSE_OCSVME_SANTINIS.eps'
# fig_save_file = '/home/dimitrios/MacroPRC11AVG_RFSE_OCSVME_7Genres.eps'

comb_lst = [
    # ['RFSE', '3Words', 'SANTINIS', 'Comb', [50000, 10000, 0.5, 100, '']],
    # ['RFSE', '3Words', 'SANTINIS', 'MinMax', [50000, 5000, 0.7, 100, '']],
    # ['RFSE', '3Words', 'SANTINIS', 'MinMax', [100000, 5000, 0.9, 100, '']],
    # ['OCSVME', '3Words', 'SANTINIS', '', [50000, 10000, 0.07, '']],
    # ['OCSVME', '3Words', 'SANTINIS', '', [50000, 10000, 0.1, '']],
    ['OCSVME', '3Words', 'SANTINIS', '', [100000, 50000, 0.07, '']],

]

plt_dsp_attr = [
    # ['k' + line_type[1] + symbol[0], 2, 14, "RFSE - Comb - $AUC$"],
    # ['k' + line_type[1] + symbol[1], 2, 14, "RFSE - Cos - $F_{1}$"],
    # ['k' + line_type[1] + symbol[2], 2, 14, "RFSE - W3G - Cos - $F_{0.5}$"],
    # ['k' + line_type[0] + symbol[0], 2, 14, "OCSVM - $AUC$"],
    # ['k' + line_type[0] + symbol[1], 2, 14, "OCSVM - $F_{1}$"],
    ['k' + line_type[0] + symbol[2], 2, 14, "OCSVM - W3G - $F_{0.5}$"],
]

leg_pos = 4

"""

# KI04
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_OCSVM_KI04_F1Based_11AVG.eps'

comb_lst = [
    ['OCSVME', '3Words', 'KI04', '', [100000, 5000, 0.3]],
    ['OCSVME', '1Words', 'KI04', '', [10000, 1000, 0.1]],
    ['OCSVME', '4Chars', 'KI04', '', [100000, 90000, 0.07]]
]

plt_dsp_attr = [
    ['k' + line_type[0] + symbol[0], 2, 14, "3W - KI04"],
    ['k' + line_type[1] + symbol[1], 2, 14, "1W - KI04"],
    ['k' + line_type[3] + symbol[3], 2, 14, "4C - KI04"]
]

leg_pos = 1


# SANTINIS
fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_OCSVM_SANTINIS_F1Based_11AVG.eps'

comb_lst = [
    ['OCSVME', '3Words', 'SANTINIS', '', [50000, 10000, 0.1]],
    ['OCSVME', '1Words', 'SANTINIS', '', [100000, 50000, 0.3]],
    ['OCSVME', '4Chars', 'SANTINIS', '', [10000, 5000, 0.5]]
]

plt_dsp_attr = [
    ['k' + line_type[6] + symbol[0], 2, 14, "3W - SANTINIS"],
    ['k' + line_type[7] + symbol[1], 2, 14, "1W - SANTINIS"],
    ['k' + line_type[8] + symbol[3], 2, 14, "4C - SANTINIS"]
]

leg_pos = 1
"""

# # # #  The Ploting Process Starts Here # # # #
fig = plt.figure(num=1, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)

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
            h5d_fl1, h5d_fl2, kfolds, params_path, binary=False, strata=None  #  binary=False <- for Micro
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
    ax.plot(
        x[non_zero_idx], y[non_zero_idx],
        plt_dsp_attr[i][0],
        linewidth=plt_dsp_attr[i][1],
        markersize=plt_dsp_attr[i][2],
        label=plt_dsp_attr[i][3]
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

# Give the poper attributes for better ploting

plt.grid(True)
# plt.legend(
#   loc='upper left', bbox_to_anchor=(0.62, 0.4), ncol=1, fancybox=True, shadow=True, fontsize=16
# )
plt.legend(loc=3, fancybox=True, shadow=True, fontsize=16)
plt.yticks(fontsize=18)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=18)
# plt.tight_layout()

# Saving the ploting to File
# plt.savefig(fig_save_file, bbox_inches='tight')

plt.show()
