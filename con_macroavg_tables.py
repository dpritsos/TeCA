# !/usr/bin/env python

import sys
import tables as tb
import numpy as np
import collections as coll
import time as tm
import json

import teca.analytics.param_combs as param_comb
from teca.data_retrieval.prconf_tables import PRConf_table
from auc_tables import params_prauc_tables


# Run for all cases
if __name__ == '__main__':

    # Parameters used for the experiments required for selecting specific or group of results
    kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    case_od = coll.OrderedDict([
        ('doc_rep', ['3Words', '1Words', '4Chars']),  # '3Words', '1Words', '4Chars'
        ('corpus', ['SANTINIS']),  # , 'SANTINIS', 'KI04', '7Genres'
        ('dist', ['']),  # '', 'MinMax', 'MIX'
        ('vocab_size', [5000, 10000, 50000, 100000]),  # 5000, 10000, 50000, 100000
        ('features_size', [500, 1000, 5000, 10000, 50000, 90000]),  # 500, 1000, 5000, 10000, 50000, 90000
        # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),  # 0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9
        ('Sigma', [0.5, 0.7, 0.9]),  # 0.5, 0.7, 0.9
        ('Iterations', [10, 50, 100]),  # 10, 50, 100
        ('KFold', [''])
    ])

    # List of all macro averaging precision recall values.
    pr_macro_lst = list()

    # Creating tables for tall the above cases.
    for case, case_path in zip(param_comb.ParamGridIter(case_od, 'list'),
                               param_comb.ParamGridIter(case_od, 'path')):

        if case[3] > case[4]:

            # Starting timing for this loop for monitoring the cosuption of the socre calculations.
            start_tm = tm.time()

            # Selecting filepath
            if case[1] == '7Genres':
                # h5d_fl = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_'
                h5d_fl = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_'

            elif case[1] == 'KI04':
                # h5d_fl = '/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_'
                h5d_fl = '/home/dimitrios/Synergy-Crawler/KI-04/RFSE_'

            else:
                # h5d_fl = '/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_'
                h5d_fl = '/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_'

            h5d_fl = h5d_fl + case[0] + '_' + case[1]

            #  Selecting files to open and setting the mix flag on/off
            mix = False
            if case[2] == 'MIX':
                h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
                h5d_fl2 = tb.open_file(h5d_fl + '_minmax.h5', 'r')
                mix = True

            elif case[2] == 'MinMax':
                h5d_fl1 = tb.open_file(h5d_fl + '_minmax.h5', 'r')
                h5d_fl2 = None

            else:
                h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
                h5d_fl2 = None

            # ### Calculating the PR Curves ###

            # Preparing input for params_prauc_tables().
            param_od = coll.OrderedDict([
                ('vocab_size', [case[3]]),
                ('features_size', [case[4]]),
                # ('nu', [case[5]]),
                ('Sigma', [case[5]]),  #
                ('Iterations', [case[6]]),  #
                ('KFold', [''])
            ])

            pr_aucz_var_table = params_prauc_tables(
                h5d_fl1, h5d_fl2, 'multiclass_macro', kfolds, param_od, mix, strata=None, trec=False
            )

            # ### Calculating Marco Averaging of Precision, Recall, F1, F0.5 ###

            # Reformationg parametera path to be given properly to PRConf_table().
            params_path = '/' + '/'.join(case_path.split('/')[4::])
            # print pr_tabel_fname

            # Calculating the Precision and Recall Scores per Genre. Precision for a Genre is...
            # ...only calculated when there is at least one sample being counted for this Genre.
            pre_cls_vect, rcl_cls_vect = PRConf_table(
                h5d_fl1, h5d_fl2, kfolds, params_path, mix, strata=None, prereccon=1
            )

            # NOTE: The Precision and Recall vectors of scores per Genre might not have the same...
            # ...leght due to Unknown_Class tag expected or not expected case or just because...
            # ...for some Class we have NO Predicitons at all.
            print pre_cls_vect, rcl_cls_vect
            macro_p = np.mean(pre_cls_vect)
            macro_r = np.mean(rcl_cls_vect)

            f1 = 2.0 * macro_p*macro_r / (macro_p+macro_r)
            if np.isnan(f1):
                f1 = 0

            f05 = 1.25 * macro_p*macro_r / (0.25*macro_p+macro_r)
            if np.isnan(f05):
                f05 = 0

            # pr_mean = (macro_pr[0]+macro_pr[1]) / 2.0

            pr_auc = pr_aucz_var_table[0, 5]  # For RFSE is 5, for OCSVME 4

            # roc_auc = roc_aucz_var_table[0, 4]  # For RFSE is 5, for OCSVME 4

            # join_auc = 1
            # for a in m_aucz_var_table[0, 4::]:  # For RFSE is 5, for OCSVME 4
            #     join_auc *= a

            # maucs_num = len(m_aucz_var_table[0, 4::])  # For RFSE is 4, for OCSVME 3
            # m_auc_f1 = maucs_num / np.sum([1.0/mauc for mauc in m_aucz_var_table[0, 3::]])
            # if np.isnan(m_auc_f1):
            #     m_auc_f1 = 0

            # Macro Averaging AUCs per Genre.
            case.extend([macro_p, macro_r, pr_auc, f05, f1])
            # m_auc_f1, join_auc, auc, pr_mean, f05, f1
            pr_macro_lst.append(case)

            # Printing the case and the time cosumed for calculation.
            print case
            timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
            print "Time elapsed : %d:%d:%d:%d" % timel

            # print pr_macro_lst

            # Saving tables
            # np.savetxt(conf_mtrx_fname, conf_mtrx)

            # pr_tabel[np.where((pr_tabel==np.NaN))] = 0.0
            # np.savetxt(pr_tabel_fname, pr_table)

            # np.set_printoptions(precision=3, threshold=10000, suppress=True, linewidth=100)
            # print conf_mtrx
            # print pr_scores[:, :]"""

            # Closing HDF5 files
            if case[2] == 'MIX':
                h5d_fl1.close()
                h5d_fl2.close()
            else:
                h5d_fl1.close()

    # Getting the macro averaged P, R, F1, F05
    pr_macroavgs = np.vstack(pr_macro_lst)

    # # # Saving Resaults to the following file.
    prnt_case_od = coll.OrderedDict([
        ('critirion_idx', [-1, -2, -3]),  # -5, -6, -8
        ('dist', ['']),
        ('corpus', ['7Genres']),  # , 'KI04', 'SANTINIS', 7Genres
        ('doc_rep', ['3Words', '1Words', '4Chars'])  # '3Words', '1Words', '4Chars'
    ])

    with open('/home/dimitrios/MaxScore_RFSE_7Genres_2016_10_11.txt', 'w') as score_sf:

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
                score_sf.write("Macro PR AUC ")
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
