#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import numpy as np
import collections as coll
from data_retrieval.rfsedata import get_predictions
from data_retrieval.rfsemixdata import get_predictions as get_predictions_mix
import base.param_combs as param_comb
import analytics.metrix as mx


#Parameters used for the experiments required for selecting specific or group of results
kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_od = coll.OrderedDict([
    ('vocab_size', [5000, 10000, 50000, 100000]),
    ('features_size', [1000, 5000, 10000, 50000, 90000]),
    #(Bagging', [0.66]),
    #('nu', [0.9]),  # [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
    ('Sigma', [0.5, 0.7, 0.9]),
    ('Iterations', [10, 50, 100])
])

case_od = coll.OrderedDict([
    ('doc_rep', ['3Words', '1Words', '4Chars']),
    ('corpus', ['SANTINIS']),  # '7Genres', 'KI04', 
    ('dist', ['', 'MinMax', 'MIX']),
    ('feat_idx', [('Feat&Voc', 1, 0), ('Sigma&Iter', 2, 3)]),
])

# Defining the file name of the experimental results to be used
#h5d_fl = str(
    # '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_3Words_7Genres'
    # '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_3Words_7Genres'
    # '/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_1Words_SANTINIS'
    # '/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_3Words_SANTINIS'
    # '/home/dimitrios/Synergy-Crawler/KI-04/RFSE_4Chars_KI04'
    # '/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_3Words_KI04'
#)

#h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
#h5d_fl2 = tb.open_file(h5d_fl + '_minmax.h5', 'r')

# Defining the directory and file name for table to be saved.
# aucz_mean_var_fname = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/tables_data/AUC_tables/AUC_1Words_KI04_MinMax_Feat&Voc.csv'
# aucz_mean_var_fname = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/tables_data/AUC_tables/AUC_3Words_7Genres_MinMax_Sigma&Iter.csv'
# _FeatVSVoc
# _Sigma&Iter


def auc_tables(h5d_fl1, h5d_fl2, params_od, mix, tbl_params, is_ttbl, strata):

    #Beginning AUC-Params table building.
    res_lst = list()

    # Loading data in a convenient form.
    for params_lst, params_path in zip(
        param_comb.ParamGridIter(params_od, 'list'),
            param_comb.ParamGridIter(params_od, 'path')):

        if params_lst[0] > params_lst[1]:

            if mix:

                pred_scores, expd_y, pred_y = get_predictions_mix(
                    h5d_fl1, h5d_fl2, kfolds, params_path, params_lst[2], gnr_num=12,
                    genre_tag=None, binary=True, strata=strata
                )

            else:

                pred_scores, expd_y, pred_y = get_predictions(
                    h5d_fl1, kfolds, params_path, genre_tag=None, binary=True, strata=strata
                )

            ###NOTE: Option is_truth_tbl is critical to be selected correctly
            #depeding on the input.
            prec, recl, t = mx.pr_curve(expd_y, pred_scores, full_curve=True, is_truth_tbl=is_ttbl)

            #With Averaging.
            #prec, recl = mx.reclev_averaging(prec, recl)

            try:
                auc_value = mx.auc(recl, prec)

                params_lst.extend([auc_value])

                res_lst.append(params_lst)

            except:
                print params_path

    #Stacking the data collected in a 2D array. Last column contain the AUC for every parameters
    #values possible combination.
    res = np.vstack(res_lst)

    #Variance Implementation.
    if tbl_params[0] == 'Feat&Voc':
        p1_lst = params_od['features_size']
        p2_lst = params_od['vocab_size']

    elif tbl_params[0] == 'Sigma&Iter':
        p1_lst = params_od['Sigma']
        p2_lst = params_od['Iterations']

    else:
        raise Exception('Requested parameters combination not been predicted in Code Design')

    p1_idx = tbl_params[1]
    p2_idx = tbl_params[2]

    #Table containing the results AUC means plus variance
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


#One Run for all cases
if __name__ == '__main__':

    for case in param_comb.ParamGridIter(case_od, 'list'):

        print case

        # Selecting filepath
        if case[1] == '7Genres':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_'

        elif case[1] == 'KI04':
            h5d_fl = '/home/dimitrios/Synergy-Crawler/KI-04/RFSE_'

        else:
            h5d_fl = '/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_'

        h5d_fl = h5d_fl + case[0] + '_' + case[1]

        # Selecting files to open and setting the mix flag on/off
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

        #Defining save file name
        aucz_mean_var_fname = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/' + \
            'Journal_IPM-Elsevier/tables_data/AUC_tables/AUC_'
        aucz_mean_var_fname = aucz_mean_var_fname + '_'.join(case[0:3]) + '_' + case[3][0] + '.csv'
        aucz_mean_var_fname = aucz_mean_var_fname.replace('__', '_Cos_')

        aucz_mean_var_table = auc_tables(
            h5d_fl1, h5d_fl2, params_od, mix, case[3], is_ttbl=True, strata=(10, 1000)
        )

        #Saving the AUC means (with variance table)
        np.savetxt(aucz_mean_var_fname, aucz_mean_var_table)

        #Closing HDF5 files
        if case[2] == 'MIX':
            h5d_fl1.close()
            h5d_fl2.close()
        else:
            h5d_fl1.close()
