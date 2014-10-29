#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import numpy as np
import collections as coll
#from data_retrieval.rfsedata import get_predictions
from data_retrieval.rfsemixdata import get_predictions
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

unknow_class = True

# Defining the file name of the experimental results to be used
h5d_fl = str(
    #'/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_4Chars_7Genres'
    #'/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_3Words_7Genres'
    #'/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_3Words_SANTINIS'
    #'/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_3Words_SANTINIS'
    '/home/dimitrios/Synergy-Crawler/KI-04/RFSE_1Words_KI04'
    #'/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_3Words_KI04'
)

h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
h5d_fl2 = tb.open_file(h5d_fl + '_minmax.h5', 'r')

#Defining the directory and file name for table to be saved.
aucz_mean_var_fname = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/tables_data/AUC_1Words_KI04_MIX_FeatVSVoc.csv'


#Beginning AUC-Params table building
res_lst = list()

#Loading data in a convenient form.
for params_lst, params_path in zip(
        param_comb.ParamGridIter(params_od, 'list'), param_comb.ParamGridIter(params_od, 'path')):

    if params_lst[0] > params_lst[1]:

        #pred_scores, expd_y, pred_y = get_predictions(
        #    h5d_fl1, kfolds, params_path, genre_tag=None, binary=True
        #)

        pred_scores, expd_y, pred_y = get_predictions(
            h5d_fl1, h5d_fl2, kfolds, params_path, params_lst[2], gnr_num=12,
            genre_tag=None, binary=True
        )

        prec, recl, t = mx.pr_curve(expd_y, pred_scores, full_curve=True)

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
p1_lst = params_od['features_size']
p2_lst = params_od['vocab_size']

#Table containing the results AUC means plus variance
aucz_mean_var_table = np.zeros((len(p1_lst), len(p2_lst)*2))

for i, p1 in enumerate(p1_lst):

    skp1 = 0

    for cc, p2 in enumerate(p2_lst):

        j = skp1 + cc

        auc_per_params = res[np.where((res[:, 1] == p1) & (res[:, 0] == p2))]

        if auc_per_params.shape[0]:
            aucz_mean_var_table[i, j] = np.mean(auc_per_params[:, -1])
            aucz_mean_var_table[i, j+1] = np.var(auc_per_params[:, -1])

        skp1 += 1

#Saving the AUC means (with variance table)
#print aucz_mean_var_table
np.savetxt(aucz_mean_var_fname, aucz_mean_var_table)

#Closing HDF5 files
h5d_fl1.close()
h5d_fl2.close()
