#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import numpy as np
import collections as coll
from data_retrieval.ocsvmedata import get_predictions as get_rfse
import base.param_combs as param_comb
import analytics.metrix as mx

  
kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_od = coll.OrderedDict( [
    ('vocab_size', [100000]), #[5000, 10000, 50000, 100000]),\
    ('features_size', [10000]), #[1000, 5000, 10000, 50000, 90000]\
    #(Bagging', [0.66]),\
    #('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),\
    ('Sigma', [0.7]), #[0.5, 0.7, 0.9]),\
    ('Iterations', [100]) #[10, 50, 100]
] )

res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_4Chars_7Genres_minmax.h5', 'r')
#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_4Chars_7Genres.h5', 'r')

#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_4Chars_SANTINIS.h5', 'r')
#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_3Words_SANTINIS.h5', 'r')

#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/RFSE_4Chars_KI04.h5', 'r')
#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_3Words_KI04.h5', 'r')



pred_y_lst = list()

exp_y_lst = list()

params_full_lst = list()

#Loading data in a convenient form.
for params_lst, params_path in zip(param_comb.ParamGridIter(params_od, 'list'), param_comb.ParamGridIter(params_od, 'path')):

    if params_lst[0] > params_lst[1]:

        rfse_data = get_rfse(res_h5file, kfolds, params_path, genre_tag=None) 

        print rfse_data

        #3rd element contain predicted y values list
        pred_y_lst.append(rfse_data[2])

        #2rd element contain predicted y values list
        exp_y_lst.append(rfse_data[1])

        params_full_lst.append(params_lst)

                 
#Stack the data collected in a 2D array.
pred_y = np.vstack(pred_y_lst)
exp_y = np.vstack(exp_y_lst)
params_full_array = np.vstack(params_full_lst)


#predicted_y = np.hstack( pred_y[ np.where(((params_full_array[:,2] == 0.5) )) ] )
#expected_y = np.hstack( exp_y[ np.where(((params_full_array[:,2] == 0.5) )) ] )
predicted_y = np.hstack( pred_y )
expected_y = np.hstack( exp_y )

conf_mtrx =  mx.contingency_table(expected_y, predicted_y, unknow_class=True, arr_type=np.int32)
conf_percent = np.divide(conf_mtrx, np.bincount(expected_y)) * 100 
pr_scores = mx.precision_recall_scores(conf_mtrx)
col_sums = conf_mtrx.sum(axis=0)

#conf_mtrx = np.array(conf_mtrx, dtype="S9")
#col_sums = np.array(col_sums, dtype="S9")

#genres = np.array([ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ])

#import numpy.lib.recfunctions as nprf
#conf_mtrx = nprf.stack_arrays((genres, conf_mtrx, col_sums), usemask=False)

conf_mtrx = np.vstack((conf_mtrx, col_sums))

np.set_printoptions(precision=3, threshold=10000, suppress=True, linewidth=100)

print conf_mtrx


print conf_percent


print pr_scores



#np.savetxt("/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/tables_data/conf_mtrx_RFSE_4Chars_7Genres.csv", conf_mtrx)
#np.savetxt("/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/tables_data/conf_percent_RFSE_4Chars_7Genres.csv", conf_percent)
#np.savetxt("/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/tables_data/pr_scores_RFSE_4Chars_7Genres.csv", pr_scores)


                                                                        
res_h5file.close()   