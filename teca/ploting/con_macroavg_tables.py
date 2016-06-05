# !/usr/bin/env python

import sys
sys.path.append('../../teca')
sys.path.append('../../../DoGSWrapper/dogswrapper')

import tables as tb
import numpy as np
import collections as coll
import base.param_combs as param_comb
from prereccon_tables import h5d_prereccon_table
from auc_tables import h5d_pr_auc_table, h5d_roc_auc_table


# One Run for all cases
if __name__ == '__main__':

    # Parameters used for the experiments required for selecting specific or group of results
    kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    case_od = coll.OrderedDict([
        ('doc_rep', ['3Words', '1Words', '4Chars']),
        ('corpus', ['7Genres', 'KI04', 'SANTINIS']),  #
        ('dist', ['', 'MinMax', 'MIX']),  # 'MinMax', 'MIX'
        ('vocab_size', [5000, 10000, 50000, 100000]),  #
        ('features_size', [500, 1000, 5000, 10000, 50000, 90000]),  #
        # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9])
        ('Sigma', [0.5, 0.7, 0.9]),  #
        ('Iterations', [10, 50, 100])  #
    ])

    # List of all macro averaging precision recall values.
    pr_macro_lst = list()

    # Creating tables for tall the above cases.
    for case, case_path in zip(param_comb.ParamGridIter(case_od, 'list'),
                               param_comb.ParamGridIter(case_od, 'path')):

        if case[3] > case[4]:

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
                h5d_fl2 = h5d_fl1

            else:
                h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
                h5d_fl2 = h5d_fl1

            # Defining save file name
            # conf_mtrx_fname = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/' + \
            #     'Journal_IPM-Elsevier/tables_data/Confusion_tables/Conf_'

            # conf_mtrx_fname = conf_mtrx_fname + '_'.join(case[0:3]) + \
            #     '_pset(' + ','.join([str(i) for i in case[3::]]) + ')'
            # conf_mtrx_fname = conf_mtrx_fname.replace('__', '_Cos_')  #  _Cos_ OCSVME
            # conf_mtrx_fname = conf_mtrx_fname.replace('0.', '0')
            # conf_mtrx_fname = conf_mtrx_fname + '.csv'

            # Reformationg parametera path to be given properly to the h5d_pre_rec_table function.
            params_path = '/' + '/'.join(case_path.split('/')[4::])
            # print pr_tabel_fname

            # Calculating the tables
            pr_table = h5d_prereccon_table(
                h5d_fl1, h5d_fl2, kfolds, params_path, mix,
                strata=None, unknow_class=True, prereccon=1
            )

            # print pr_table
            # 0/0

            # Preparing input for h5d_auc_table().
            param_od = coll.OrderedDict([
                ('vocab_size', [case[3]]),
                ('features_size', [case[4]]),
                # ('nu', [case[5]])
                ('Sigma', [case[5]]),  #
                ('Iterations', [case[6]])  #
            ])

            pr_aucz_var_table = h5d_pr_auc_table(
                h5d_fl1, h5d_fl2, kfolds, param_od, mix, is_ttbl=False, strata=None, trec=False
            )

            # roc_aucz_var_table = h5d_roc_auc_table(
            #     h5d_fl1, h5d_fl2, kfolds, param_od, mix, is_ttbl=True, strata=None
            # )

            # This is Marco AUC where the Genre are not considered as Truth Table Case...
            # ...that is every AUC for every Genre is different then they all are macro-averaged.
            # m_aucz_var_table = h5d_auc_table(
            #     h5d_fl1, h5d_fl2, kfolds, param_od, mix, is_ttbl=False, strata=None
            # )

            # Calculating Marco Averaging of Precision and Recall.
            pr_table[np.where(np.isnan(pr_table))] = 0  # Only for OCSVME
            macro_pr = np.mean(pr_table, axis=0)
            macro_pr[np.where(np.isnan(macro_pr))] = 0

            f1 = 2.0 * macro_pr[0]*macro_pr[1] / (macro_pr[0]+macro_pr[1])
            if np.isnan(f1):
                f1 = 0

            f05 = 1.25 * macro_pr[0]*macro_pr[1] / (0.25*macro_pr[0]+macro_pr[1])
            if np.isnan(f05):
                f05 = 0

            # pr_mean = (macro_pr[0]+macro_pr[1]) / 2.0

            pr_auc = pr_aucz_var_table[0, 4]  # For RFSE is 4, for OCSVME 3

            # roc_auc = roc_aucz_var_table[0, 4]  # For RFSE is 4, for OCSVME 3

            # join_auc = 1
            # for a in m_aucz_var_table[0, 4::]:  # For RFSE is 4, for OCSVME 3
            #     join_auc *= a

            # maucs_num = len(m_aucz_var_table[0, 4::])  # For RFSE is 4, for OCSVME 3
            # m_auc_f1 = maucs_num / np.sum([1.0/mauc for mauc in m_aucz_var_table[0, 3::]])
            # if np.isnan(m_auc_f1):
            #     m_auc_f1 = 0

            # Macro Averaging AUCs per Genre.
            case.extend([macro_pr[0], macro_pr[1], pr_auc, f05, f1])
            # m_auc_f1, join_auc, auc, pr_mean, f05, f1
            pr_macro_lst.append(case)

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

    # Getting macro averaged P, R, F1, F05
    pr_macroavgs = np.vstack(pr_macro_lst)

    prnt_case_od = coll.OrderedDict([
        ('critirion_idx', [-1, -2, -3]),  # -5, -6, -8
        ('dist', ['', 'MinMax', 'MIX']),  # 'MinMax', 'MIX'
        ('corpus', ['7Genres', 'KI04', 'SANTINIS']),
        ('doc_rep', ['3Words', '1Words', '4Chars'])
    ])

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

        print

        if idx == -1:
            print "F1"
        elif idx == -2:
            print "F0.5"
        elif idx == -3:
            print "PR AUC"
        elif idx == -4:
            print "ROC AUC"
        elif idx == -5:
            print "Mean"
        elif idx == -6:
            print "Marco JoinP AUCs / GeoMean"
        elif idx == -7:
            print "Marco AUCs JP F1"
        elif idx == -8:
            print "Precision"

        # Printing Optimal.
        values_float_array = np.array(selctd_rows[:, idx], dtype=np.float64)
        opt_idx = np.argmax(values_float_array)
        print selctd_rows[opt_idx, :]
