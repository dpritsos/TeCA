#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import analytics.metrix as mx
import data_retrieval.rfsedata as data
import base.param_combs as param_comb
import collections as coll
import numpy as np


kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_od = coll.OrderedDict( [
    ('vocab_size', [5000, 10000, 50000, 100000]), #[5000, 10000, 50000, 100000]),\
    ('features_size', [500, 1000, 5000, 10000, 50000, 90000]),\
    #'3.Bagging', [0.66],\
    ('Sigma', [0.5, 0.7, 0.9]),\
    ('Iterations', [10, 50, 100])
] )


#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Char4Grams-Koppels-Bagging_method_kfolds-10_GridSearch_TEST.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_TEST_NOBAGG.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TEST_NOBAGG.h5', 'r')

#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Words_RFSE.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Words3Grams_RFSE.h5', 'r')
res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_Char4Grams_SANTINIS.h5', 'r')


symbol = [ "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+" ]
line_type = [ "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-" ]
color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']


parms_lst = list()
tpr_lst = list()
fpr_lst = list()

#Loading data in a convenient form.
for params_lst, params_path in zip(param_comb.ParamGridIter(params_od, 'list'), param_comb.ParamGridIter(params_od, 'path')):

    if params_lst[0] > params_lst[1]:

        ps, ey = data.get_predictions(res_h5file, kfolds, params_path, genre_tag=None)

        tpr, fpr, t = mx.roc_curve(ey, ps, full_curve=True)

        tpr_lst.append(tpr)
        fpr_lst.append(fpr)
        parms_lst.append(params_lst)
        

#Stack the data collected in a 2D array. Last column contain the AUC for every parameters values possible combination.
parms_arr = np.vstack(parms_lst)
tpr_arr = np.vstack(tpr_lst)
fpr_arr = np.vstack(fpr_lst)

plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
#sub3 = plt.ylim(ymin=0, ymax=12000)

#Variance Implementation.
for voc_size in params_od['vocab_size']: 
  
    x = list()
    y = list()
    yerr = list()
    xerr = list()

    #for voc_size in params_od['vocab_size']: 

        #idxs = np.where((parms_arr[:,0] == voc_size) & (parms_arr[:,1] == feat_size))
    idxs = np.where(parms_arr[:,0] == voc_size)

    idxs = idxs[0]

    if idxs.shape[0]:
        print np.mean(tpr_arr[idxs,:], axis=0)
        y = np.mean(tpr_arr[idxs,:], axis=0)
        yerr = np.var(tpr_arr[idxs,:], axis=0)
        x = np.mean(fpr_arr[idxs,:], axis=0)
        xerr = np.var(fpr_arr[idxs,:], axis=0)

    plt.plot(x, y, label=voc_size)
    #plt.errorbar(x, y, xerr, label=voc_size)

#plt.xticks(params_od['vocab_size'])
plt.grid()
plt.legend(loc=0) #bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)

plt.show()



"""


plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')







i = 0

for params_lst, params_path in zip(param_comb.ParamGridIter(params_od, 'list'), param_comb.ParamGridIter(params_od, 'path')):

    if params_lst[0] > params_lst[1]:

        PS, EY = get_predictions(res_h5file, kfolds, params_path, genre_tag=None)

        X, Y, T = roc_curve(EY, PS, full_curve=True)

        #Y, X = recl_lv_P_avg(Y, X)       
        
        i += 1

        plt.plot(X, Y, color[i] + line_type[i] + symbol[i], linewidth=1, markeredgewidth=0.01, label="("+str(i)+") Feat "+str(params_lst[0])  )
        
        plt.grid(True)
        plt.legend(loc=4, fancybox=True, shadow=True)    
        #plt.legend(loc=3, bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)    
       
plt.show()
                
                """                                                         
res_h5file.close()   