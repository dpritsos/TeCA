# !/usr/bin/env python

import sys
import tables as tb
import numpy as np
import collections as coll

sys.path.append('../../teca')
sys.path.append('../../../DoGSWrapper/dogswrapper')

from data_retrieval.data import multiclass_res
from data_retrieval.data import rfse_multiclass_multimeasure_res
from data_retrieval.data import rfse_onevsall_res
from data_retrieval.data import rfse_onevsall_multimeasure_res
import base.param_combs as param_comb
import analytics.metrix as mx


def params_prauc_tables(h5d_fl1, h5d_fl2, curvetype, kfolds,
                        params_od, mix, strata, trec, unknown_class=True):
    """Area Under the Curve(AUC) paired with table of parameters for PR curve.

        # # #  Make proper Definition here # # #

    """
    # Selecting whether the resaults should be retured in binary(i.e. Trueth-Table)...
    # ...or multi-class value form.
    if curvetype == 'multiclass':
        binary = True
    else:
        binary = False

    # Beginning AUC-Params table building.
    res_lst = list()

    #  Loading data in a convenient form.
    for params_lst, params_path in zip(
        param_comb.ParamGridIter(params_od, 'list'),
            param_comb.ParamGridIter(params_od, 'path')):

        # Defining list for AUC values storage. For this loop.
        auc_values = list()

        if params_lst[0] > params_lst[1]:

            if mix:

                pred_scores, expd_y, pred_y = rfse_multiclass_multimeasure_res(
                    h5d_fl1, h5d_fl2, kfolds, params_path, binary=binary, strata=strata
                )

            else:

                pred_scores, expd_y, pred_y = multiclass_res(
                    h5d_fl1, kfolds, params_path, binary=binary, strata=strata
                )

            # NOTE: Crossckecking and replacing the class-tags of the experiment to virtual...
            # ...class tags refering to the index of the np.unique(expd_y) vector in order...
            # ...to ease the calculations of the curves.
            tags2idx_ref = np.unique(expd_y)
            i_fix = 0
            if tags2idx_ref[0] > 0:
                i_fix = 1
            for i, tg in enumerate(tags2idx_ref):
                expd_y[np.where(expd_y == tg)] = i + i_fix
                pred_y[np.where(pred_y == tg)] = i + i_fix

            # Selecting the case and calculating the precision recall curves.
            if curvetype == 'multiclass':

                # NOTE: Option 'is_truth_tbl' is critical to be selected correctly depending...
                # ...on the input.
                prec, recl, t = mx.pr_curve(
                    expd_y, pred_scores, full_curve=True, is_truth_tbl=True
                )

                # Interpolated at 11-Recall-Levels.
                prec, recl = mx.reclev11_max(prec, recl, trec=trec)

            elif curvetype == 'multiclass_macro':

                # NOTE: Option 'unknow_class' is critical to be selected correctly depending...
                # ...on the input.
                prec, recl, t = mx.pr_curve_macro(
                    expd_y, pred_y, pred_scores, full_curve=True, unknown_class=unknown_class
                )

                # Interpolated at 11-Recall-Levels.
                prec, recl = mx.reclev11_max(prec, recl, trec=trec)

            elif curvetype == 'onevsall':

                # Finding unique genres.
                gnr_tgs = np.unique(expd_y)

                # Precsion and Recall scores lists of the PR curve per genre.
                prec_lst = list()
                recl_lst = list()

                # Calculating AUC per genre tag.
                for gnr in gnr_tgs:

                    if mix:

                        pred_scores, expd_y, pred_y = onevsall_multimeasure_res(
                            h5d_fl1, h5d_fl2, gnr, kfolds, params_path
                        )

                    else:

                        pred_scores, expd_y, pred_y = onevsall_res(
                            h5d_fl1, gnr, kfolds, params_path
                        )

                    # NOTE: Option 'is_truth_tbl' is critical to be selected correctly depending...
                    # ...on the input.
                    prec_val, recl_val, t = mx.pr_curve(
                        expd_y, pred_scores, full_curve=True, is_truth_tbl=False
                    )

                    # Interpolated at 11-Recall-Levels.
                    prec_val, recl_val = mx.reclev11_max(prec_val, recl_val, trec=trec)

                    # Keeping Precsion and Recall scores of the PR curve per genre.
                    prec_lst.append(prec)
                    recl_lst.append(recl)

                # Calculating the PR Averaged Macro Curves values for 1-vs-All case.
                prec = np.mean(np.vstack(prec_lst), axis=0)
                recl = np.mean(np.vstack(recl_lst), axis=0)

            else:
                raise Exception('Invalide curvetype argument value.')

            # Saving the AUC value and extending parameters list with AUC(s).
            try:
                params_lst.extend([mx.auc(recl, prec)])
            except:
                print "Warning:", params_path, "PR AUC is for these params has set to 0.0"
                params_lst.extend([0.0])

            # Appending the parameters list together with their respective AUC(s).
            res_lst.append(params_lst)

    # Stacking and returning the data collected in a 2D array. Last column contain the AUC for...
    # ...every parameters values possible combination.
    return np.vstack(res_lst)
