# !/usr/bin/env python

import sys
import tables as tb
import numpy as np
import collections as coll

sys.path.append('../../teca')
sys.path.append('../../../DoGSWrapper/dogswrapper')

from data_retrieval.rfsedata import get_predictions
from data_retrieval.rfsemixdata import get_predictions as get_predictions_mix
import base.param_combs as param_comb
import analytics.metrix as mx


def h5d_auc_table(h5d_fl1, h5d_fl2, kfolds, params_od, mix, is_ttbl, strata, trec):
    """Area Under the Curve(AUC) paired with table of parameters

        # # #  Make proper Definition here # # #

    """

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

                pred_scores, expd_y, pred_y = get_predictions_mix(
                    h5d_fl1, h5d_fl2, kfolds, params_path, params_lst[2],
                    genre_tag=None, binary=is_ttbl, strata=strata
                )

            else:

                pred_scores, expd_y, pred_y = get_predictions(
                    h5d_fl1, kfolds, params_path, genre_tag=None, binary=is_ttbl, strata=strata
                )

            if is_ttbl:

                # NOTE:Option is_truth_tbl is critical to be selected correctly depending...
                # ...on the input.
                prec, recl, t = mx.pr_curve(
                    expd_y, pred_scores, full_curve=True, is_truth_tbl=is_ttbl
                )

                # Interpolated at 11-Recall-Levels.
                prec, recl = mx.reclev11_max(prec, recl, trec=trec)

                try:
                    auc_values.append(mx.auc(recl, prec))
                except:
                    print "Warning:", params_path, "AUC is for these params has set to 0.0"
                    auc_values.append(0.0)

            else:

                # Finding unique genres.
                gnr_tgs = np.unique(expd_y)

                # Calculating AUC per genre tag.
                for gnr in gnr_tgs:

                    # Converting expected Y to binary format.
                    expd_y_bin = np.where((expd_y == gnr), 1, 0)

                    # NOTE:Option is_truth_tbl is critical to be selected correctly depending...
                    # ...on the input.
                    prec, recl, t = mx.pr_curve(
                        expd_y_bin, pred_scores, full_curve=True, is_truth_tbl=is_ttbl
                    )

                    # Interpolated at 11-Recall-Levels.
                    prec, recl = mx.reclev11_max(prec, recl, trec=trec)

                    try:
                        auc_values.append(auc(recl, prec))
                    except:
                        print "Warning:", params_path, "AUC is for these params has setted to 0.0"
                        auc_values.append(0.0)

            # Extending parameters list with AUC(s).
            params_lst.extend(auc_values)

            # Appending the parameters list together with their respective AUC(s).
            res_lst.append(params_lst)

    # Stacking and returning the data collected in a 2D array. Last column contain the AUC for...
    # ...every parameters values possible combination.

    return np.vstack(res_lst)


def h5d_auc_table_2d(h5d_fl1, h5d_fl2, kfolds, params_od, mix, tbl_params, is_ttbl, strata):
    """Area Under the Curve(AUC) 2D table of parameters

        # # #  Make proper Definition here # # #

    """
    # Calling the h5f_auc_table for calculating the AUC for the given paramters and H5D files.
    res = h5d_auc_table(h5d_fl1, h5d_fl2, kfolds, params_od, mix, is_ttbl, strata)

    # Variance Implementation.
    if tbl_params[0] == 'Feat&Voc':
        p1_lst = params_od['features_size']
        p2_lst = params_od['vocab_size']

    elif tbl_params[0] == 'Sigma&Iter':
        p1_lst = params_od['Sigma']
        p2_lst = params_od['Iterations']

    elif tbl_params[0] == 'Nu&Feat':
        p1_lst = params_od['nu']
        p2_lst = params_od['features_size']

    else:
        raise Exception('Requested parameters combination not been predicted in Code Design')

    p1_idx = tbl_params[1]
    p2_idx = tbl_params[2]

    # Table containing the results AUC means plus variance.
    aucz_mean_var_table = np.zeros((len(p1_lst), len(p2_lst)*2))

    for i, p1 in enumerate(p1_lst):

        skp1 = 0

        for cc, p2 in enumerate(p2_lst):

            j = skp1 + cc

            auc_per_params = res[np.where((res[:, p1_idx] == p1) & (res[:, p2_idx] == p2))]

            if auc_per_params.shape[0]:
                aucz_mean_var_table[i, j] = np.mean(auc_per_params[:, -1])
                aucz_mean_var_table[i, j+1] = np.var(auc_per_params[:, -1])

            skp1 += 1

    return aucz_mean_var_table


# One Run for all cases.
if __name__ == '__main__':

    # Parameters used for the experiments required for selecting specific or group of results.
    kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    params_od = coll.OrderedDict([
        ('vocab_size', [10000]),  # 5000, 10000, 50000, 100000
        ('features_size', [5000]),  # 1000, 5000, 10000, 50000, 90000
        ('nu', [0.5]),  # 0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9
        # ('Sigma', [0.9]), # 0.5, 0.7, 0.9
        # ('Iterations', [100]) # 10, 50, 100
    ])

    case_od = coll.OrderedDict([
        ('doc_rep', ['3Words', '1Words', '4Chars']),
        ('corpus', ['SANTINIS']),  # , 'SANTINIS', '7Genres', 'KI04'
        ('dist', ['']),  # , 'MinMax', 'MIX'
        # ('feat_idx', [('Feat&Voc', 1, 0), ('Sigma&Iter', 2, 3)]),
        # ('feat_idx', [('Nu&Feat', 2, 1)])
    ])

    # Creating tables for tall the above cases.
    for case in param_comb.ParamGridIter(case_od, 'list'):

        print case

        #  Selecting filepath
        if case[1] == '7Genres':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_'

        elif case[1] == 'KI04':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_'

        else:
            h5d_fl = '/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_'

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
        aucz_var_fname = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/' + \
            'Journal_IPM-Elsevier/tables_data/AUC_tables/AUC_'
        aucz_var_fname = aucz_var_fname + '_'.join(case[0:3]) + '.csv'
        aucz_var_fname = aucz_var_fname.replace('_.', '_OCSVME.')  # _Cos., _OCSVME.

        aucz_var_table = h5d_auc_table(
            h5d_fl1, h5d_fl2, kfolds, params_od, mix, is_ttbl=True, strata=None
            # (10, 1000)
        )

        # aucz_var_table = h5d_auc_table_2d(
        #     h5d_fl1, h5d_fl2, kfolds, params_od, mix,
        #     tbl_params=case_od['feat_idx'][0], is_ttbl=True, strata=None
        #     # (10, 1000)
        # )

        # Saving the AUC means (with variance table).
        # np.savetxt(aucz_var_fname, aucz_var_table)

        np.set_printoptions(precision=3, threshold=10000, suppress=True, linewidth=100)
        # print aucz_var_table[np.argmax(aucz_var_table[:, 3])]
        print aucz_var_table

        # Closing HDF5 files
        if case[2] == 'MIX':
            h5d_fl1.close()
            h5d_fl2.close()
        else:
            h5d_fl1.close()
