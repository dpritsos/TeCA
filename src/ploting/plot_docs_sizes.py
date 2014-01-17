#!/usr/bin/env python

import tables as tb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
from curve_pr_rfse import zero_class_dist, zclass_dist_per_class, prcurve, ZClass_DocSize
from sklearn import grid_search

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

#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Words_RFSE_Part-1n2.h5', 'r')
res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Char4Grams_RFSE.h5', 'r')


color_pallet2 = { 500:['k']*24, 1000:['r']*24, 5000:['g']*24, 10000:['b']*24, 50000:['y']*24, 90000:['m']*24 }
color_pallet = { 5000:['k']*24, 10000:['r']*24, 50000:['g']*24, 100000:['b']*24 }
#[, , , , ['c']*24, ['m']*24, ['y']*24]
symbol = [ "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+" ]
line_type = [ "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-" ]

plt.figure(num=None, figsize=(22, 12), dpi=80, facecolor='w', edgecolor='k')

Zero_Dist_lst = list()

plot_cnt = 0
last_voc_size = '5000'

for i, params in enumerate([100000]): #grid_search.IterGrid(params_range)):

    #params_path = "/".join( [ key.split('.')[1] + str(value).replace('.','') for key, value in sorted( params.items() ) ] )
    params_path = '/vocab_size100000' #+ params_path
    
    X_len, Y_len = ZClass_DocSize(res_h5file, kfolds, params_path, genre_tag=None)
    plt.barh(X_len, Y_len)
    

    plt.grid(True)
          

plt.show()
                                                                         
res_h5file.close()   