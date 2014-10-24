#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import numpy as np
import collections as coll
from data_retrieval.rfsedata import get_predictions as get_rfse
from data_retrieval.rfsemixdata import get_predictions
import base.param_combs as param_comb
import analytics.metrix as mx

# Parameters used for the experiments required for selecting specific or
# group of results
kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


params_od = coll.OrderedDict([
    ('vocab_size', [100000]),  # [5000, 10000, 50000, 100000]),
    ('features_size', [10000]),  # [1000, 5000, 10000, 50000, 90000]
    #(Bagging', [0.66]),
    #('nu', [0.9]),  # [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
    ('Sigma', [0.7]),  # [0.5, 0.7, 0.9]),
    ('Iterations', [100])  # [10, 50, 100]
])

unknow_class = True

# Defining the file name of the experimental results to be used
h5d_fl = str(
    #/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_3Words_7Genres'
    #'/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_3Words_7Genres'
    '/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_3Words_SANTINIS'
    #'/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_3Words_SANTINIS'
    #'/home/dimitrios/Synergy-Crawler/KI-04/RFSE_1Words_KI04_minmax'
    #'/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_3Words_KI04'
)

h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
h5d_fl2 = tb.open_file(h5d_fl + '_minmax.h5', 'r')

# Defining the directory and file name for table to be saved
conf_matrix_fname = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/ECIR2015/tables_data/Confusion_tables/Conf_MinMax_4Chars_SANTINIS_PSet(09,100,100K,5K).csv'

# Beginning Contingency table building
pred_y_lst, exp_y_lst, params_full_lst = (list(), list(), list())

# Loading data in a convenient form.
for params_lst, params_path in \
    zip(param_comb.ParamGridIter(params_od, 'list'),
        param_comb.ParamGridIter(params_od, 'path')):

    if params_lst[0] > params_lst[1]:

        rfse_data = get_rfse(h5d_fl1, kfolds, params_path, genre_tag=None)
        #rfse_data = get_predictions(
        #    h5d_fl1, h5d_fl2, kfolds, params_path, params_lst[2], gnr_num=12, genre_tag=None
        #)

        # 3rd element contain predicted y values list
        pred_y_lst.append(rfse_data[2])

        # 2rd element contain predicted y values list
        exp_y_lst.append(rfse_data[1])

        params_full_lst.append(params_lst)

# Stack the data collected in a 2D array.
pred_y = np.hstack(pred_y_lst)
exp_y = np.hstack(exp_y_lst)

# Here is actually built the contingency table.
conf_mtrx = mx.contingency_table(exp_y, pred_y, unknow_class=unknow_class, arr_type=np.int32)
col_sums = conf_mtrx.sum(axis=0)
#conf_mtrx = np.vstack((conf_mtrx, col_sums))
#conf_percent = np.divide(conf_mtrx, np.bincount(expected_y)) * 100
pr_scores = mx.precision_recall_scores(conf_mtrx)

np.set_printoptions(precision=3, threshold=10000, suppress=True, linewidth=100)

print conf_mtrx
print pr_scores[:, :]

# Saving the AUC means (with variance table)
#np.savetxt(conf_matrix_fname, conf_mtrx)

# Closing HDF5 file
h5d_fl1.close()
h5d_fl2.close()
