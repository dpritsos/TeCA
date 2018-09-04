# !/usr/bin/env python

import sys
import tables as tb
import numpy as np
import collections as coll
import time as tm
import json

import teca.analytics.param_combs as param_comb
from teca.data_retrieval.data import svm_onevsall_scores
from prconf_tables import PRConf_table
from auc_tables import params_prauc_tables


# Run for all cases
if __name__ == '__main__':

    # Parameters used for the experiments required for selecting specific or group of results
    kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    genres = ["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
    # genres = [
    #  "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop"
    # ]

    case_od = coll.OrderedDict([
        ('doc_rep', ['C4G']),
        ('corpus', ['7Genres']),  # , 'SANTINIS', 'KI04', '7Genres'
        ('svm_type', ['binary']),
        ('vocab_size', [100000]),
        ('features_size', [1000, 5000, 10000]),
        # ('l', [0.3, 0.8]),
        # ('c1_w', [0.3, 0.7]),
        # ('c2_w', [0.3, 0.7]),
        # ('mrgn_nw', [0.3, 0.7]),
        # ('mrgn_fw', [0.3, 0.7]),
        ('mrgn_nw', [0.05, 0.15]),
        ('mrgn_fw', [0.05, 0.15]),
        ('onlytest_gnrs_splts', [1, 2, 3, 4, 5, 6, 7]),
        ('onlytest_splt_itrs', [0, 1, 2, 3]),
        ('kfolds', ['']),
    ])

    # List of all macro averaging precision recall values.
    prf1_lst = list()

    # Creating tables for tall the above cases.
    for case, case_path in zip(param_comb.ParamGridIter(case_od, 'list'),
                               param_comb.ParamGridIter(case_od, 'path')):

        if case[3] > case[4]:

            # Starting timing for this loop for monitoring the cosuption of the socre calculations.
            start_tm = tm.time()

            # Selecting filepath
            if case[1] == '7Genres':
                # h5d_fl = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_'
                h5d_fl = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/LOPSVM_BIN_Openness_C4G_7Genres_22/LOPSVM_BIN_Openness_'

            elif case[1] == 'KI04':
                # h5d_fl = '/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_'
                h5d_fl = '/home/dimitrios/Synergy-Crawler/KI-04/Openness_RFSE_W3G_KI04/Openness_RFSE_'

            else:
                # h5d_fl = '/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_'
                h5d_fl = '/home/dimitrios/Synergy-Crawler/SANTINIS/Openness_RFSE_'

            h5d_fl = h5d_fl + case[0] + '_' + case[1]

            #  Selecting files to open and setting the mix flag on/off
            mix = False

            h5d_fl1 = tb.open_file(h5d_fl + '_2.2.h5', 'r')
            h5d_fl2 = None

            # ### Calculating Precision, Recall, F1, F0.5 ###

            # Reformationg parametera path to be given properly to PRConf_table().
            params_path = '/' + '/'.join(case_path.split('/')[3::])
            print params_path
            p, r, f1 = svm_onevsall_scores(h5d_fl1, genres, kfolds, params_path)

            # Macro Averaging AUCs per Genre.
            case.extend([p, r, f1])
            prf1_lst.append(case)

            # Printing the case and the time cosumed for calculation.
            print case
            timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
            print "Time elapsed : %d:%d:%d:%d" % timel

            h5d_fl1.close()

    # Getting the macro averaged P, R, F1, F05
    prf1_arr = np.vstack(prf1_lst)

    print prf1_arr

    0/0
    # # # Saving Resaults to the following file.
    prnt_case_od = coll.OrderedDict([
        ('critirion_idx', [-1, -2, -3]),  # -5, -6, -8
        ('dist', ['', 'MinMax', 'MIX']),  # '', 'MinMax',
        ('corpus', ['KI04']),  # , 'KI04', 'SANTINIS', 7Genres
        ('doc_rep', ['W3G'])
    ])

    with open('/home/dimitrios/Openess_KI04_W3G.txt', 'w') as score_sf:

        for idx, dm, cr, dr in param_comb.ParamGridIter(prnt_case_od, 'list'):

            # Getting Combination.
            selctd_rows = pr_macroavgs[
                np.where(
                    (pr_macroavgs[:, 0] == dr) &
                    (pr_macroavgs[:, 1] == cr) &
                    (pr_macroavgs[:, 2] == dm)
                )
            ]

            # Printing... for f1 = -1 idx, f05 = -2 idx, auc = -4 idx

            if idx == -1:
                score_sf.write("F1 ")
            elif idx == -2:
                score_sf.write("F0.5 ")
            elif idx == -3:
                score_sf.write("PR MACRO AUC ")
            elif idx == -4:
                score_sf.write("ROC AUC ")
            elif idx == -5:
                score_sf.write("Mean ")
            elif idx == -6:
                score_sf.write("Marco JoinP AUCs / GeoMean ")
            elif idx == -7:
                score_sf.write("Marco AUCs JP F1 ")
            elif idx == -8:
                score_sf.write("Precision ")

            # Printing Optimal.
            values_float_array = np.array(selctd_rows[:, idx], dtype=np.float64)
            try:
                opt_idx = np.argmax(values_float_array)
                json.dump(fp=score_sf, obj=list(selctd_rows[opt_idx, :]))
                score_sf.write("\n")
            except:
                score_sf.write("Empty Array\n")
                print "Empty Array"
