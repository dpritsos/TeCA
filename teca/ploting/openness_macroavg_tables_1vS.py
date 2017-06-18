# !/usr/bin/env python

import sys
import tables as tb
import numpy as np
import collections as coll
import time as tm
import json
sys.path.append('../../teca')
sys.path.append('../../../DoGSWrapper/dogswrapper')

import base.param_combs as param_comb
from prconf_tables import PRConf_table
from auc_tables import params_prauc_tables
from data_retrieval.data import svm_onevsall_scores
import analytics.metrix as mx


# Run for all cases
if __name__ == '__main__':

    # Parameters used for the experiments required for selecting specific or group of results
    kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    case_od = coll.OrderedDict([
        ('svm_type', ['oneclass']),
        ('vocab_size', [100000]),
        ('features_size', [1000, 5000]),  # , 10000
        ('nu', [0.08, 0.1, 0.18, 0.3, 0.5, 0.7, 0.9]),
        ('l', [0.3, 0.8]),
        ('c1_w', [0.3, 0.7]),
        ('c2_w', [0.3, 0.7]),
        ('mrgn_nw', [0.3, 0.7]),
        ('mrgn_fw', [0.3, 0.7]),
        ('onlytest_gnrs_splts', [1, 2, 3, 4, 5, 6]),
        ('onlytest_splt_itrs', [0, 1]),
        # ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ('kfolds', ['']),
    ])

    # Openning the Results HD5 file.
    h5d_fl = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/LOPSVM_Openness_W1G_KI04_OCSVM_MyWay/LOPSVM_W1G_KI04_minmax_F1.h5', 'r')

    # List of all macro averaging precision recall values.
    pr_macro_lst = list()

    # Creating tables for tall the above cases.
    for case, case_path in zip(param_comb.ParamGridIter(case_od, 'list'),
                               param_comb.ParamGridIter(case_od, 'path')):

        start_tm = tm.time()

        # ######## Calculating the PR Curves ########
        pscores, expd_y, pred_y = svm_onevsall_scores(h5d_fl, case_path, kfolds, trd=np.Inf)

        # NOTE: Option 'unknow_class' is critical to be selected correctly depending...
        # ...on the input.
        prec, recl, t = mx.pr_curve_macro(
            expd_y, pred_y, pscores, full_curve=True,
        )

        # Interpolated at 11-Recall-Levels.
        prec, recl = mx.reclev11_max(prec, recl, trec=False)  # <----terc?

        # Saving the AUC value and extending parameters list with AUC(s).
        try:
            macro_auc = mx.auc(recl, prec)
        except:
            print "Warning:", params_path, "PR Macro-AUC is for these params has set to 0.0"
            macro_auc = 0.0

        # ######## Calculating Marco Averaging of Precision, Recall, F1, F0.5 ########

        # Getting the expected classes.
        exp_cls_tags_set = np.unique(expd_y)

        # Calculating contigency table.
        conf_mtrx = mx.seq_contingency_table(
            expd_y, pred_y, exp_cls_tags_set=exp_cls_tags_set, arr_type=np.int32
        )

        # Getting the number of samples per class. Zero tag is inlcuded.
        smpls_per_cls = np.bincount(np.array(expd_y, dtype=np.int))

        # Keeping from 1 to end array in case the expected class tags start with above zero values.
        if smpls_per_cls[0] == 0 and exp_cls_tags_set[0] > 0:
            smpls_per_cls = smpls_per_cls[1::]
        elif smpls_per_cls[0] > 0 and exp_cls_tags_set[0] == 0:
            pass  # same as --> smpls_per_cls = smpls_per_cls
            # Anythig else should rase an Exception.
        else:
            raise Exception("Samples count in zero bin is different to the expected class tag cnt!")

        # Calculating Precision per class.
        precisions_vect = [
            dg / float(pred_docs)
            for dg, pred_docs in zip(np.diag(conf_mtrx), np.sum(conf_mtrx, axis=1))
            if pred_docs > 0
        ]

        # Calculating Recall per class.
        recalls_vect = [
            dg / float(splpc)
            for dg, splpc in zip(np.diag(conf_mtrx), smpls_per_cls)
            if splpc > 0
        ]

        # This funciton works only for the mx.contingency_table() output.
        # pr_tbl = mx.precision_recall_scores(conf_mtrx)

        # NOTE: The Precision and Recall vectors of scores per Genre might not have the same...
        # ...leght due to Unknown_Class tag expected or not expected case or just because...
        # ...for some Class we have NO Predicitons at all.
        macro_p = np.mean(precisions_vect)
        macro_r = np.mean(recalls_vect)

        f1 = 2.0 * macro_p*macro_r / (macro_p+macro_r)
        if np.isnan(f1):
            f1 = 0

        f05 = 1.25 * macro_p*macro_r / (0.25*macro_p+macro_r)
        if np.isnan(f05):
            f05 = 0

        # Keeping the scores respectively to the paramter paths.
        case.extend([macro_p, macro_r, macro_auc, f05, f1])
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

        # Saving table
        with open('/home/dimitrios/Openness_1vsSet_OCSVM_MyWay_2016_11_21.txt', 'w') as score_sf:
            json.dump(fp=score_sf, obj=np.array(case).tolist())

    # Closing HDF5 files
    h5d_fl.close()

    # Getting the macro averaged P, R, F1, F05
    pr_macroavgs = np.vstack(pr_macro_lst)
    print pr_macroavgs

    """
    # # # Saving Resaults to the following file.
    prnt_case_od = coll.OrderedDict([
        ('critirion_idx', [-1, -2, -3]),  # -5, -6, -8
        ('dist', ['', 'MinMax', 'MIX']),  # '', 'MinMax',
        ('corpus', ['KI04']),  # , 'KI04', 'SANTINIS', 7Genres
        ('doc_rep', ['W3G'])
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
        """
