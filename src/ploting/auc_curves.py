#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import numpy as np
import collections as coll
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import data_retrieval.rfsedata as data
import base.param_combs as param_comb
import analytics.metrix as mx
from sklearn import grid_search

from sklearn import grid_search


kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_od = coll.OrderedDict( [
    ('vocab_size', [5000, 10000, 50000, 100000]),\
    ('features_size', [500, 1000, 5000, 10000, 50000, 90000]),\
    #'3.Bagging', [0.66],\
    ('Sigma', [0.5, 0.7, 0.9]),\
    ('Iterations', [100]) #, 50, 100])
] )

#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Char4Grams-Koppels-Bagging_method_kfolds-10_GridSearch_TEST.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_TEST_NOBAGG.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TEST_NOBAGG.h5', 'r')

res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Words_RFSE.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Words3Grams_RFSE.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Char4Grams_RFSE.h5', 'r')



color_pallet2 = { 500:['k']*24, 1000:['r']*24, 5000:['g']*24, 10000:['b']*24, 50000:['y']*24, 90000:['m']*24 }
color_pallet = { 5000:['k']*24, 10000:['r']*24, 50000:['g']*24, 100000:['b']*24 }
#[, , , , ['c']*24, ['m']*24, ['y']*24]
symbol = [ "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+" ]
line_type = [ "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-" ]



res_lst = list()

#Loading data in a convenient form.
for params_lst, params_path in zip(param_comb.ParamGridIter(params_od, 'list'), param_comb.ParamGridIter(params_od, 'path')):

    if params_lst[0] > params_lst[1]:

        ps, ey = data.get_predictions(res_h5file, kfolds, params_path, genre_tag=None)

        p, r, t = mx.pr_curve(ey, ps, full_curve=False)

        try:
            auc_value = mx.auc(r, p)

            params_lst.extend([auc_value])

            res_lst.append(params_lst)
        
        except:
            
            print params_path
            

#Stack the data collected in a 2D array. Last column contain the AUC for every parameters values possible combination.
res = np.vstack(res_lst)


plt.figure(num=None, figsize=(22, 12), dpi=80, facecolor='w', edgecolor='k')
#sub3 = plt.ylim(ymin=0, ymax=12000)


#Variance Implementation.
for voc_size in params_od['features_size']: #params_od['vocab_size']:
  
    x = list()
    y = list()
    yerr = list()

    for feat_size in params_od['vocab_size']: #params_od['features_size']:

        auc_per_sigma = res[ np.where((res[:,0] == feat_size) & (res[:,1] == voc_size)) ]

        if auc_per_sigma.shape[0]:
            print auc_per_sigma[:,-1]
            x.append(feat_size)    
            y.append( np.mean(auc_per_sigma[:,-1]) )
            yerr.append( np.var(auc_per_sigma[:,-1]) )

    plt.errorbar(x, y, yerr, label=voc_size)

plt.xticks(params_od['vocab_size'])
plt.grid()
plt.legend(loc=0) #bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)

plt.show()




"""        

        #color = color_pallet[ params['1.vocab_size'] ]
        color = color_pallet2[ params['2.features_size'] ]

        plot_cnt += 1

        a = auc(X, Y)



        sub1.plot(plot_cnt, a, symbol[i] + line_type[i] + color[i], linewidth=1, markeredgewidth=2, label="("+str(plot_cnt)+") RFSE"+params_path  )
      
        sub1.grid(True)
        #sub2.grid(True)
        sub1.legend(loc=3, bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)    
       

#sub3.boxplot(Zero_Dist_lst)

#plt.savefig('/home/dimitrios/Desktop/Expected_ZClass.pdf')

plt.show()
"""
                                                                         
res_h5file.close()   