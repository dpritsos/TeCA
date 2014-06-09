#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
from analytics.metrix import  pr_curve, roc_curve, reclev_averaging, reclev_max_around, reclev_nearest, smooth_linear
from data_retrieval.rfsedata import get_predictions
from sklearn import grid_search
from sklearn.metrics import roc_curve as roc
import base.param_combs as param_comb
import collections as coll
import numpy as np
from sklearn.metrics import precision_recall_curve

kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_od = coll.OrderedDict( [
    ('vocab_size', [100000]), #[5000, 10000, 50000, 100000]),\
    ('features_size', [1000, 5000, 10000, 50000, 90000]), #1000, 5000, 10000, 50000, 90000]\
    #'3.Bagging', [0.66],\
    #('nu', [0.05]) #, 0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9])
    ('Sigma', [0.5]), #[0.5, 0.7, 0.9])\
    ('Iterations', [100]) #[10, 50, 100]
] )

#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_4Chars_7Genres.h5', 'r')
#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_4Chars_7Genres.h5', 'r')

res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_4Chars_SANTINIS_SQRD.h5', 'r')
#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVME_4Chars_SANTINIS.h5', 'r')

#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/RFSE_1Words_KI04_SQRD.h5', 'r')

symbol = [ "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+" ]
line_type = [ "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-" ]
color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
color2 = ['r', 'g', 'b', 'k', 'c', 'y', 'm', 'r', 'g', 'b', 'k', 'c', 'y', 'm']

fg1 = plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
ax1 = fg1.add_subplot(111)
fg2 = plt.figure(num=2, figsize=(30, 8), dpi=80, facecolor='w', edgecolor='k')
ax2 = fg2.add_subplot(111)

i = 0
bar_width = 0.15

for params_lst, params_path in zip(param_comb.ParamGridIter(params_od, 'list'), param_comb.ParamGridIter(params_od, 'path')):

    if params_lst[0] > params_lst[1]:

        PS, EY, PR_Y = get_predictions(res_h5file, kfolds, params_path, genre_tag=None)

        Y, X, T = pr_curve(EY, PS, full_curve=True) #roc(EY, PS) #roc_curve(EY, PS, full_curve=False) #
        #Y, X, T = precision_recall_curve(EY, PS) #<== They are not 100% percent the same.
        
        #Smoothing out the Precision (Y axis) of the P-R Curve.
        #CRITICAL: The P sould be inverted from the lowest to 
        #the highest values*. 
        # *( it suppose the higest values to be normally fist in order )
        #Y = smooth_linear(Y[::-1])
        
        #Inverting the Y (i.e. Precition) axis values after has been smoothed out.
        #Y = Y[::-1]

        # OR
        
        #Y, X = smooth_linear(Y[::-1], X[::-1])
        #Y, X = Y[::-1], X[::-1]

        Y, X = reclev_averaging(Y, X)

        #Y, X = reclev_nearest(Y, X)

        #Y, X = reclev_max_around(Y, X)

        

        #plt.locator_params(nbins=4)

        ax1.plot(X, Y, color[i] + line_type[i] + symbol[i], linewidth=1, markeredgewidth=1, label="("+str(i)+") Feat "+str(params_lst[1])  )
        #ax1.title("SANTINIS")
        #ax1.yticks([ .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.00]) #.10, .15, .20, .25, .30, .35, .40, .45,
        ax1.grid(True)
        ax1.legend(loc=4, fancybox=True, shadow=True)    
        #ax1.legend(loc=3, bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)

        hist, bins = np.histogram(PR_Y, 12)

        #print len(PR_Y[PR_Y == 0]), hist, bins
        
        #tmp = np.zeros(len(bins+1))
        #tmp[0:-1] = hist[::]
        #hist = tmp

        ax2.bar(bins[0:12] + bar_width , hist, width=0.1, color=color2[i]) #i, zdir='y', alpha=0.8, align='center', width=bins[1]-bins[0])  
        i += 1 
        bar_width += 0.1 

plt.xticks(bins[0:12] + bar_width, ('Dont Know', "blog", "eshop", "faq", "frontpage", "listing", "php", "spage", "diy_mini", "editorial", "feat_articles", "short_bio"))

plt.show()
                                                                         
res_h5file.close()   