#!/usr/bin/env python

import sys
sys.path.append('../../src')

import json
import pickle as pkl
import numpy as np
import tables as tb

from sklearn import grid_search
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
 
from analytics.docmetrix import get_idx2gnr, Docs_Sizes, get_docsizes

kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_range = {
    '1.vocab_size': [5000], #5000, 10000, 50000, 100000],
    '2.features_size': [500, 1000, 5000, 10000, 50000, 90000], #[500, 1000, 5000, 10000, 50000, 90000],
    #'3.Bagging': [0.66],
    '4.Iterations': [100], #[10, 50, 100],
    '3.Sigma': [0.9] #0.5, 0.7, 0.9]
} 

root_path_Word1G = '/home/dimitrios/Synergy-Crawler/SANTINIS/Kfolds_Vocs_Inds_Word_1Grams/'
root_path_Char4G = '/home/dimitrios/Synergy-Crawler/SANTINIS/Kfolds_Vocs_Inds_Char_4Grams/'

#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Char4Grams-Koppels-Bagging_method_kfolds-10_GridSearch_TEST.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_TEST_NOBAGG.h5', 'r')
#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TEST_NOBAGG.h5', 'r')

#res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Words_RFSE.h5', 'r')
res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Char4Grams_RFSE.h5', 'r')


color_pallet2 = { 500:['k']*24, 1000:['r']*24, 5000:['g']*24, 10000:['b']*24, 50000:['y']*24, 90000:['m']*24 }
color_pallet = { 5000:['k']*24, 10000:['r']*24, 50000:['g']*24, 100000:['b']*24 }
#[, , , , ['c']*24, ['m']*24, ['y']*24]
symbol = [ "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+", "^", "*", "x", "+", "*", "^", "x", "+" ]
line_type = [ "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-", "--", "--", "--", "--", "-" , "-", "-", "-" ]

#plt.figure(num=None, figsize=(22, 12), dpi=80, facecolor='w', edgecolor='k')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

for i, params in enumerate(grid_search.IterGrid(params_range)):

    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')

    if params['1.vocab_size'] > params['2.features_size']:

        params_path = "/".join( [ key.split('.')[1] + str(value).replace('.','') for key, value in sorted( params.items() ) ] )
        params_path = '/' + params_path
        
        #color = color_pallet[ params['1.vocab_size'] ]
        color = color_pallet2[ params['2.features_size'] ]

        #fidx2gnr = get_idx2gnr(root_path_Char4G, 'Corpus_filename_shorted.lst')
        #X_sets, Y_sets, Z_sets = Docs_Sizes(res_h5file, kfolds, params_path, root_path_Char4G, fidx2gnr)

        doc_sizes_kf_lst = get_docsizes(res_h5file, params['1.vocab_size'], kfolds=[0]) #range(10))
        
        #for zi, (x, y, z) in enumerate(zip(X_sets, Y_sets)):
        #for zi, y in enumerate(doc_sizes_kf_lst):
        y = doc_sizes_kf_lst
        #hist, bins = np.histogram(Y_len, bins=500)
        #print hist, bins
        hist, bins = np.histogram(y, 500)
        #print hist, bins
        tmp = np.zeros(len(bins+1))
        tmp[0:-1] = hist[::]
        hist = tmp

        ax.bar(bins, hist, i, zdir='y', alpha=0.8, align='center', width=bins[1]-bins[0])
        #ax.bar(bins[0:100], hist[0:100], zi, zdir='y', alpha=0.8, align='center', width=bins[1]-bins[0])
        
        #ax.set_ylim3d(0,11) 
        #ax.set_xlim3d(0,60000) 
        #plt.yticks(range(12), Z_sets, size="small")
        #plt.title('False Negative("Don''t Know" excluded) Document-Size-Distributions per Genre (Char4Grams)')
        
        #plt.title('Full Corpus Document-Size-Distributions per Genre (Char4Grams)')
        
        #plt.title('"Don''t Know"-(Should Have Been) Document-Size-Distributions per Genre (Char4Grams)')
        #plt.title('Missclassified (Excluding "Don''t Know") Document-Size-Distributions per Genre (Char4Grams)')
        plt.title( str(params['2.features_size']) )

        #ax.w_yaxis.set_ticklabels(Z_sets, size="small") 
        
        #sub1.legend(loc=3, bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)    
       
        #ax.xaxis.set_ticks(np.arange(0,300010,20000))
        plt.setp(plt.xticks()[1], rotation=45, size='small')
        #plt.setp(plt.zticks()[1], rotation=45, size='small')
        plt.setp(plt.yticks()[1], size='small')

#plt.savefig('/home/dimitrios/Desktop/Expected_ZClass.pdf')
ax.set_xlabel('# Terms')
ax.set_zlabel('# Documents')
ax.set_ylabel('Genres')

plt.show()
                                                                         
res_h5file.close()