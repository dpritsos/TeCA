#!/usr/bin/env python

import sys
sys.path.append('../../src')

import tables as tb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
from analytics.curve_pr_rfse import  prcurve
from analytics.docmetrix import zero_class_dist, zclass_dist_per_class, ZClass_DocSize
from sklearn import grid_search

import analytics.pr_curves_to_11_standard_recall_levels as srl

kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_range = {
    '1.vocab_size' : [100000], #5000, 10000, 50000, 
    '2.features_size' : [500, 1000, 5000, 10000, 50000, 90000], #[500, 1000, 5000, 10000, 50000, 90000],
    #'3.Bagging' : [0.66],
    '4.Iterations' : [100], #[10, 50, 100],
    '3.Sigma' : [0.5, 0.7, 0.9]
} 

#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Char4Grams-Koppels-Bagging_method_kfolds-10_GridSearch_TEST.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_TEST_NOBAGG.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TEST_NOBAGG.h5', 'r')

res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Words_RFSE.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Char4Grams_RFSE.h5', 'r')



color_pallet2 = { 500:['k']*24, 1000:['r']*24, 5000:['g']*24, 10000:['b']*24, 50000:['y']*24, 90000:['m']*24 }
color_pallet = { 5000:['k']*24, 10000:['r']*24, 50000:['g']*24, 100000:['b']*24 }
#[, , , , ['c']*24, ['m']*24, ['y']*24]
symbol = [ "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+" ]
line_type = [ "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-" ]

plt.figure(num=None, figsize=(22, 12), dpi=80, facecolor='w', edgecolor='k')

gs = gspec.GridSpec(5, 3)

sub1 = plt.subplot(gs[:-1, 0])
sub2 = plt.subplot(gs[:-1, 1])
sub3 = plt.subplot(gs[:-1, 2])
#sub3 = plt.ylim(ymin=0, ymax=12000)


Zero_Dist_lst = list()

plot_cnt = 0
last_voc_size = '5000'


singleplot = True

for i, params in enumerate(grid_search.IterGrid(params_range)):

    if params['1.vocab_size'] > params['2.features_size']:

        params_path = "/".join( [ key.split('.')[1] + str(value).replace('.','') for key, value in sorted( params.items() ) ] )
        params_path = '/' + params_path
        
        X, Y, mark_X, mark_Y = prcurve(res_h5file, kfolds, params_path, genre_tag=None)

        Zero_Dist_lst.append( zero_class_dist(res_h5file, kfolds, params_path, genre_tag=None) )

        #Zero_Dist_lst.append( zclass_dist_per_class(res_h5file, kfolds, params_path, genre_tag=None) )

        #color = color_pallet[ params['1.vocab_size'] ]
        color = color_pallet2[ params['2.features_size'] ]

        plot_cnt += 1

        sub1.plot(X, Y, symbol[i] + line_type[i] + color[i], linewidth=1, markeredgewidth=2, label="("+str(plot_cnt)+") RFSE"+params_path  )
        
        print Y
        
        A, B = srl.interpol_soothed_pr_curve(Y[::-1], X)

        sub2.plot(B, A, symbol[i] + line_type[i] + color[i], linewidth=1, markeredgewidth=2, label="("+str(plot_cnt)+") RFSE"+params_path  )

        A, B = srl._interpol_soothed_pr_curve(Y[::-1], X)

        sub3.plot(B, A, symbol[i] + line_type[i] + color[i], linewidth=1, markeredgewidth=2, label="("+str(plot_cnt)+") RFSE"+params_path  )
        
        if mark_X !=None:
            plt.plot(mark_X, mark_Y, color[i] + symbol[i], markeredgewidth=15)

        #if  singleplot:
        #    X_len, Y_len = ZClass_DocSize(res_h5file, kfolds, params_path, genre_tag=None)
        #    sub2.bar(X_len, Y_len)
        #    #sub2.plot(X_len, Y_len,  symbol[i] + color[i], linewidth=1, markeredgewidth=2)
        #    singleplot = False

        sub1.grid(True)
        #sub2.grid(True)
        sub1.legend(loc=3, bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)    
       

#sub3.boxplot(Zero_Dist_lst)

#plt.savefig('/home/dimitrios/Desktop/Expected_ZClass.pdf')

plt.show()
                                                                         
res_h5file.close()   