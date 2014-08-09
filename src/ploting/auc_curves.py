#!/usr/bin/env python

import sys
sys.path.append('../../src')
sys.path.append('../../../DoGSWrapper/src')

import tables as tb
import numpy as np
import collections as coll
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
from data_retrieval.rfsedata import get_predictions as get_rfse
import base.param_combs as param_comb
import analytics.metrix as mx
from sklearn import grid_search

from sklearn import grid_search

"""
from matplotlib import rc, rcParams
rc('font', **{'family':'serif', 'serif':['Helvetica'],
              'monospace':['Computer Modern Typewriter']})
params = {'backend': 'eps',
      'text.latex.preamble': [r"\usepackage{upgreek}",
                              r"\usepackage{siunitx}",
                              r"\usepackage{amsmath}",
                              r"\usepackage{amstext}",],
      'axes.labelsize': 18,
      #'axes.linewidth': 1,
      #'text.fontsize':17,
      'legend.fontsize': 16,
      'xtick.labelsize': 16,
      #'xtick.major.width' : 0.75,
      'ytick.labelsize': 16,
      'figure.figsize': [8.8,6.8],
      #'figure.dpi': 120,
      'text.usetex': True,
      'axes.unicode_minus': True,
      'ps.usedistiller' : 'xpdf'
      }          
rcParams.update(params)
rcParams.update({'figure.autolayout':True})
"""

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

  
kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_od = coll.OrderedDict( [
    ('vocab_size', [5000, 10000, 50000, 100000]),\
    ('features_size', [1000, 5000, 10000, 50000, 90000]), #1000, 5000, 10000, 50000, 90000]\
    #'3.Bagging', [0.66],\
    #('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),\
    ('Sigma', [0.5, 0.7, 0.9]),\
    ('Iterations', [10, 50, 100]) #[10, 50, 100]
] )

#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/RFSE_4Chars_7Genres_minmax.h5', 'r')
#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_1Words_7Genres.h5', 'r')

res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_4Chars_SANTINIS_minmax.h5', 'r')
#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/SANTINIS/OCSVM_3Words_SANTINIS.h5', 'r')

#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/RFSE_4Chars_KI04.h5', 'r')
#res_h5file = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/OCSVM_3Words_KI04.h5', 'r')

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/auc_curves_vocabVSfeat_KI04_4Chars.eps'


symbol = [ 'o', 'v', '^', '+', 'x', 's', '*', '<', '>', 'H', '1', '2', '3', '4', 'D', 'h', '8', 'd', 'p', '.', ',' ]

line_type = [ '--', '--', '--', '--', '-', '-', '-', '-', '-', '--', '--', '--', '--.', '-' , '-', '-', '-', '--', '--', '--', '--']

color = [ 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']

color2 = ['r', 'g', 'b', 'k', 'c', 'y', 'm', 'r', 'g', 'b', 'k', 'c', 'y', 'm']


res_lst = list()

#Loading data in a convenient form.
for params_lst, params_path in zip(param_comb.ParamGridIter(params_od, 'list'), param_comb.ParamGridIter(params_od, 'path')):

    if params_lst[0] > params_lst[1]:

        pred_scores, expd_y, pred_y = get_rfse(res_h5file, kfolds, params_path, genre_tag=None)

        prec, recl, t = mx.pr_curve(expd_y, pred_scores, full_curve=True)

        try:
            auc_value = mx.auc(recl, prec)

            params_lst.extend([auc_value])

            res_lst.append(params_lst)
        
        except:
            
            print params_path
            

#Stack the data collected in a 2D array. Last column contain the AUC for every parameters values possible combination.
res = np.vstack(res_lst)

plt.figure(num=None, figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
#sub3 = plt.ylim(ymin=0, ymax=12000)

i = 0;

#Variance Implementation.
for param_1 in params_od['vocab_size']: 
    
    i += 1
    x = list()
    y = list()
    yerr = list()

    for param_2 in params_od['features_size']: 

        if param_2 > 10000: continue;

        auc_per_sigma = res[ np.where((res[:,1] == param_2) & (res[:,0] == param_1)) ]

        if auc_per_sigma.shape[0]:
            #print auc_per_sigma[:,-1]
            x.append(param_2)    
            y.append( np.mean(auc_per_sigma[:,-1]) )
            yerr.append( np.var(auc_per_sigma[:,-1]) )

    plt.errorbar(x, y, yerr, fmt= color[i] + line_type[i] + symbol[i], linewidth=1, markeredgewidth=2, label=param_1)

    """ fmt='-', ecolor=None, elinewidth=None, capsize=3,
         barsabove=False, lolims=False, uplims=False,
         xlolims=False, xuplims=False, errorevery=1,
         capthick=None)"""

#plt.xticks(params_od['features_size'])
plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000])

#plt.xticks(np.arange(0,100000,5000))
#plt.xticks(np.arange(0,120,10))

plt.yticks(np.arange(0.4,1.05,0.05))


plt.grid()
plt.legend(loc=4) #bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)
plt.tight_layout()

plt.savefig(fig_save_file, bbox_inches='tight')

plt.show()
                                                                        
res_h5file.close()   