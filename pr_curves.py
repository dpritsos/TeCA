

# !/usr/bin/env python

import tables as tb
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import sys

from teca.analytics.metrix import pr_curve, pr_curve_macro, reclev11_max
from teca.data_retrieval.data import multiclass_res
from teca.data_retrieval.data import rfse_multiclass_multimeasure_res
from teca.data_retrieval.data import rfse_multiclass_multimeasure_res2
from teca.data_retrieval.data import rfse_onevsall_res
from teca.data_retrieval.data import rfse_onevsall_multimeasure_res

# Funciton for creating the parameter paths requred for get_predicitons()
def plist2ppath(params_lst, ensbl='RFSE'):

    if ensbl == 'RFSE':

        pnames = [
            'terms_type', 'vocab_size', 'features_size', 'sim_func',
            'Sigma', 'Iterations', 'marked_uknw_ctg_lst', 'kfolds'
        ]

        return ''.join(['/' + n + str(v).replace('.', '') for n, v in zip(pnames, params_lst)])

    if ensbl == 'RFSE-COS-OLD':

        pnames = [
            'vocab_size', 'features_size', 'Sigma', 'Iterations', 'KFold'
        ]

        return ''.join(['/' + n + str(v).replace('.', '') for n, v in zip(pnames, params_lst)])

    elif ensbl == 'OCSVME-OLD':

        pnames = [
            'vocab_size', 'features_size', 'nu', 'KFold'
        ]

        return ''.join(['/' + n + str(v).replace('.', '') for n, v in zip(pnames, params_lst)])

    elif ensbl == 'OCSVME':

        pnames = [
            'terms_type', 'vocab_size', 'features_size', 'nu', 'marked_uknw_ctg_lst', 'kfolds'
        ]

        return ''.join(['/' + n + str(v).replace('.', '') for n, v in zip(pnames, params_lst)])

    elif ensbl == 'NNDR-GENS':

        pnames = [
            'terms_type', 'vocab_size',
            'dims', 'min_trm_fq', 'win_size', 'algo', 'alpha', 'min_alpha', 'epochs', 'decay',
            'split_ptg', 'ukwn_slt_ptg', 'rt_lims_stp', 'lmda',
            'marked_uknw_ctg_lst', 'kfolds'
        ]

        return ''.join(['/' + n + str(v).replace('.', '') for n, v in zip(pnames, params_lst)])


    elif ensbl == 'NNDR-TF':

        pnames = [
            'terms_type', 'vocab_size', 'split_ptg', 'ukwn_slt_ptg',
            'rt_lims_stp', 'lmda', 'marked_uknw_ctg_lst', 'kfolds'
        ]

        return ''.join(['/' + n + str(v).replace('.', '') for n, v in zip(pnames, params_lst)])

    else:
        raise Exception('Invalid Ensebmle Name')


kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 7Genres
fig_save_file = '/home/dimitrios/RESULTS/' +\
                'NNDR_W3G_Best_RFSE-OCSVME-Baselines_Short.eps'

comb_lst = [

    [
        'OCSVME-OLD',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/' +\
        'OCSVM_3Words_SANTINIS.h5',
        [50000, 10000, 0.1, '']
    ],
    [
        'RFSE-COS-OLD',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/' +\
        'RFSE_3Words_SANTINIS.h5',
        [100000, 10000, 0.5, 100, '']
    ],
    [
        'NNDR-TF',
        '/home/dimitrios/RESULTS/NNDR_TF_RAW/' +\
        'OpenSet_MU_NNDR_W3G_TF_15-2-3_V5-50-100k_SANTINIS_2018_10_07.h5',
        ['W3G', 10000, 0.5, 0.5, '[0.4, 1.0, 0.2]', 1.5, 12, '']
    ],
    [
        'NNDR-GENS',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/Open_W3G_SANTINIS/' +\
        'OpenSet_MarkedUknown_NNRD_W3G_Gensim_SANTINIS_2018_08_17.h5',
        ['W3G', 'NA', 100, 3, 3, 'PV-DBOW', 0.025, 0.025, 10, 0.02, 0.5, 0.5,
            '[0.8, 1.0, 0.2]', 0.5, 12, '']
    ],

]




"""
    [
        'OCSVME-OLD',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/' +\
        'OCSVM_4Chars_SANTINIS.h5',
        [10000, 5000, 0.5, '']
    ],
    [
        'RFSE-COS-OLD',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/' +\
        'RFSE_4Chars_SANTINIS.h5',
        [50000, 1000, 0.5, 100, '']
    ],
    [
        'NNDR-TF',
        '/home/dimitrios/RESULTS/NNDR_TF_RAW/' +\
        'OpenSet_MarkedUknown_NNRD_C4G_V5000_SANTINIS_2018_09_22.h5',
        ['C4G', 5000, 0.7, 0.3, '[0.8, 1.0, 0.2]', 0.5, 12, '']
    ],
    [
        'NNDR-GENS',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/Open_C4G_SANTINIS/' +\
        'OpenSet_MarkedUknown_NNRD_C4G_Gensim_SANTINIS_2018_08_17.h5',
        ['C4G', 'NA', 50, 3, 8, 'PV-DBOW', 0.025, 0.025, 10, 0.002, 0.5, 0.5, '[0.8, 1.0, 0.2]', 0.5, 12, '']
    ]

"""


"""

    [
        'OCSVME-OLD',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/' +\
        'OCSVM_1Words_SANTINIS.h5',
        [50000, 5000, 0.1, '']
    ],
    [
        'RFSE',
        '/mnt/turnstone/SANTINIS/TF_W1G_SANTINIS/' +\
        'RFSE_W1G_V50000_SANTINIS_2018_02_22_Moreiters_FIXED.h5',
        ['W3G', 50000, 10000, 'cosine_sim', 0.5, 1000, 12, '']
    ],

    [
        'NNDR-TF',
        '/home/dimitrios/RESULTS/NNDR_TF_RAW/' +\
        'OpenSet_MarkedUknown_NNRD_W1G_V5000_SANTINIS_2018_09_22.h5',
        ['W1G', 5000, 0.7, 0.5, '[0.8, 1.0, 0.2]', 0.2, 12, '']
    ],

    [
        'NNDR-GENS',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/Open_W1G_SANTINIS/' +\
        'OpenSet_MarkedUknown_NNRD_W1G_Gensim_SANTINIS_2018_08_17.h5',
        ['W1G', 'NA', 50, 3, 3, 'PV-DBOW', 0.025, 0.025, 10, 0.02, 0.5, 0.5, '[0.8, 1.0, 0.2]', 0.5, 12, '']
    ]

"""

"""

    [
        'OCSVME-OLD',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/' +\
        'OCSVM_3Words_SANTINIS.h5',
        [50000, 10000, 0.1, '']
    ],
    [
        'RFSE-COS-OLD',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/' +\
        'RFSE_3Words_SANTINIS.h5',
        [100000, 10000, 0.5, 100, '']
    ],
    [
        'NNDR-TF',
        '/home/dimitrios/RESULTS/NNDR_TF_RAW/' +\
        'OpenSet_MU_NNDR_W3G_TF_15-2-3_V5-50-100k_SANTINIS_2018_10_07.h5',
        ['W3G', 10000, 0.5, 0.5, '[0.4, 1.0, 0.2]', 1.5, 12, '']
    ],
    [
        'NNDR-GENS',
        '/home/dimitrios/Synergy-Crawler/SANTINIS/Open_W3G_SANTINIS/' +\
        'OpenSet_MarkedUknown_NNRD_W3G_Gensim_SANTINIS_2018_08_17.h5',
        ['W3G', 'NA', 100, 3, 3, 'PV-DBOW', 0.025, 0.025, 10, 0.02, 0.5, 0.5,
            '[0.8, 1.0, 0.2]', 0.5, 12, '']
    ],

"""



plt_dsp_attr = [
    ['grey', '--', '+', "OCSVM - TF"],
    ['black', '--', '^', "RFSE - TF"],
    ['blue', '-', 'o', "NNDR - TF"],
    ['red', '-', '*', "NNDR - DF"],
]


# # # #  The Ploting Process Starts Here # # # #
fig = plt.figure(num=1, figsize=(12, 8), facecolor='w', edgecolor='k')  # dpi=300,
ax = fig.add_subplot(111)

annots_lst = list()
labels_lst = list()

for i, comb_val in enumerate(comb_lst):

    h5d_fname = comb_val[1]
    h5d_fl = tb.open_file(h5d_fname, 'r')

    # Getting the predictions
    if comb_val[2][3] == 'combo':

        # Building the parapmeters path
        params_path = plist2ppath(comb_val[2], ensbl=comb_val[0])
        print params_path

        pred_scores, expd_y, pred_y = rfse_multiclass_multimeasure_res2(
            h5d_fl, ['cosine_sim', 'minmax_sim'],
            kfolds, params_path, binary=False, strata=None
        )

        """
        pred_scores, expd_y, pred_y = rfse_multiclass_multimeasure_res(
            # h5d_fl, h5d_fl2, kfolds, params_path, comb_val[4][2],
            # genre_tag=None, binary=True, strata=None
            h5d_fl, h5d_fl2, kfolds, params_path, binary=False, strata=None
            #  binary=False <- for Micro
        )
        """

    else:

        # Building the parapmeters path
        params_path = plist2ppath(comb_val[2], ensbl=comb_val[0])
        print params_path

        pred_scores, expd_y, pred_y = multiclass_res(
            h5d_fl, kfolds, params_path, binary=False, strata=None
        )

    # Closing the h5d files.
    h5d_fl.close()

    # Creating the Actual PRC.
    # y, x, t = pr_curve(expd_y, pred_scores, full_curve=True, is_truth_tbl=True)

    # Creating the Actual MACRO PRC.
    y, x, t, srz = pr_curve_macro(
        expd_y, pred_y, pred_scores, full_curve=True
    )

    # y, x, srz = y[4::], x[4::], srz[4::]

    # print y, x,

    # for ii, ss in enumerate(srz):
    #    print ii, ss

    # Getting the max 11 Recall Leves in TREC way.
    # if i == 0:
    y, x = reclev11_max(y, x, trec=False)
    # print
    # print y, x

    # Selecting array indices with non-zero cells.
    non_zero_idx = np.where(y > 0)

    # # # Do the Plotting
    linestyle = {
        "color": plt_dsp_attr[i][0],
        "linestyle": plt_dsp_attr[i][1],
        "marker": plt_dsp_attr[i][2],
        "linewidth": 2,
        # "markeredgewidth": 6,
        'markeredgecolor': plt_dsp_attr[i][0],
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
plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=12)
plt.yticks(np.arange(0.60, 1.05, 0.05), fontsize=12)
plt.ylabel('Precision', fontsize=14)
plt.xlabel('Recall', fontsize=14)
# plt.tight_layout()

# Saving the ploting to File
plt.savefig(fig_save_file, bbox_inches='tight')

plt.show()
