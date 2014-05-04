#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
from analytics.metrix import  pr_curve, recl_lv_P_avg, roc_curve, std_avg_pr_curve, close_to_11_rl, nrst_smooth_pr
from data_retrieval.rfsedata import get_predictions
from sklearn import grid_search
from sklearn.metrics import roc_curve as roc
import base.param_combs as param_comb
import collections as coll


kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_od = coll.OrderedDict( [
    ('vocab_size', [100000]), #[5000, 10000, 50000, 100000]),\
    ('features_size', [1000, 5000, 10000, 50000, 90000]), #1000, 5000, 10000, 50000, 90000]\
    #'3.Bagging', [0.66],\
    ('Sigma', [0.5]), #[0.5, 0.7, 0.9])\
    ('Iterations', [100]) #[10, 50, 100]
] )


#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Words_RFSE.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Words3Grams_RFSE.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/Kfolds_Vocs_Inds_Word_1Grams_New_Sqrd.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_3Words_7Genres.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_Char4Grams_Sqrd.h5', 'r')

res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/RFSE_4Chars_KI04.h5', 'r')


symbol = [ "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+" ]
line_type = [ "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-" ]
color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']

plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

i = 0

for params_lst, params_path in zip(param_comb.ParamGridIter(params_od, 'list'), param_comb.ParamGridIter(params_od, 'path')):

    if params_lst[0] > params_lst[1]:

        PS, EY = get_predictions(res_h5file, kfolds, params_path, genre_tag=None)

        Y, X, T = pr_curve(EY, PS, full_curve=True) #roc(EY, PS) #roc_curve(EY, PS, full_curve=False) #

        Y, X = recl_lv_P_avg(Y, X)   #std_avg_pr_curve, close_to_11_rl, recl_lv_P_avg, nrst_smooth_pr
        
        i += 1

        #plt.locator_params(nbins=4)

        plt.plot(X, Y, color[i] + line_type[i] + symbol[i], linewidth=1, markeredgewidth=1, label="("+str(i)+") Feat "+str(params_lst[1])  )
        plt.title("SANTINIS")
        plt.yticks([.10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.00])
        plt.grid(True)
        plt.legend(loc=4, fancybox=True, shadow=True)    
        #plt.legend(loc=3, bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)    
       
plt.show()
                                                                         
res_h5file.close()   