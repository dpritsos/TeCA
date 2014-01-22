#!/usr/bin/env python

import json
import pickle as pkl
import numpy as np
import tables as tb

from sklearn import grid_search
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec




def Idx_2_Genre(root_path, corpus_lst_fname):

    with open(root_path + corpus_lst_fname, 'r')  as f:
        flst = json.load(f)
        gnr_lst = [ fpth.split('/')[5] for fpth in flst ]

    return gnr_lst


def Cross_Val_Indx(root_path, corpus_lst_fname):

    with open(root_path + corpus_lst_fname, 'r')  as f:
        crv_idxs_lst = json.load(f)

    return crv_idxs_lst



def Docs_Sizes(res_h5file, kfolds, params_path, root_path, fidx2gnr=None):

    #Listed on:
    # Exepcted Y
    # Predicted Y
    # Predicted Score
    
    EY_lst = list()
    PY_lst = list()
    DCL_lst = list()
    PS_lst = list()
    crv_idxs_lst = list()

    prmpath = '/' + params_path.split('/')[1] 
    doc_len = res_h5file.getNode(prmpath+'/KFold'+str(0), name='docs_term_counts' ).read()
    DCL_lst.append( doc_len[0::] )
    
    for k in kfolds:

        #pc_per_iter = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_classes_per_iter').read()
        #gnr_pred_cnt = np.where(pc_per_iter == genre_tag, 1, 0) 
        #fold_ps = np.sum(gnr_pred_cnt, axis=0) / np.float(pc_per_iter.shape[0])   
        #PS_lst.append(fold_ps) 

        #Loading Expected_Y indices for this fold --- temp 0 fold
        with open(root_path + 'kfold_crv_' + str(k) + '.idx', 'r')  as f:
            crv_idxs = json.load(f)
        crv_idxs_lst.append(crv_idxs)
        
        #pred_scores = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_scores' ).read()
        #PS_lst.append( pred_scores )
        
        exp_y = res_h5file.getNode(params_path+'/KFold'+str(k), name='expected_Y' ).read()
        EY_lst.append( exp_y ) #[0:-1000] )

        pre_y = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_Y' ).read()
        PY_lst.append( pre_y ) #[0:-1000] )

    #EY_lst.append( exp_y[1000::] )
    #PY_lst.append( pre_y[1000::] )

    #Make Tables for Ensemble Algo
    EY = np.hstack(EY_lst)
    PY = np.hstack(PY_lst)
    DCL = np.hstack(DCL_lst)
    CRV = np.hstack(crv_idxs_lst)
    #PS = np.hstack(PS_lst)

    #print DCL.shape

    #Short Results by Cross Validation Idices
    crv_idx_idx = np.argsort(CRV)
    
    #Short Results by Predicted Scores
    #inv_srd_idx = np.argsort(EY)[:]

    PY = PY[ crv_idx_idx ]
    EY = EY[ crv_idx_idx ]
    CRV = CRV[ crv_idx_idx ]

    u_CRV, uCRV_i = np.unique(CRV, return_index=True)

    PY = PY[ uCRV_i ]
    EY = EY[ uCRV_i ]
    CRV = CRV[ uCRV_i ]
    
    Y = DCL[ (PY != EY ) & (PY == 0) ]

    constrain_inds = CRV #[ (PY == EY) & (PY == 0) ]
    
    if fidx2gnr:

        #Cutting X_len in Genre Sets
        Y_sets = list()
        y_set = list()
        X_sets = list()

        last_gnr = fidx2gnr[0]
        Z_sets = list()
        Z_sets.append(last_gnr)

       for crv_i in CRV:
            
            gname = fidx2gnr[crv_i]

            if gname != last_gnr: 
                last_gnr = gname 
                Z_sets.append(last_gnr)
                Y_sets.append(y_set)    
                X_sets.append( range(len(y_set)) )
                print len(y_set)
                y_set = []
           
            if crv_i in constrain_inds:
                y_set.append( Y[crv_i] )

            else:
                pass #y_set.append( -10 )
        Y_sets.append(y_set)   
        print len(y_set) 
        X_sets.append( range(len(y_set)) )

        Y = Y_sets
        X = X_sets
        Z = Z_sets

    else:
        X = [range( Y.shape[0] )]
        Y = [Y]
        Z = [None]

    return X, Y, Z



if __name__ == '__main__':

    kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    params_range = {
        '1.vocab_size' : [100000], #5000, 10000, 50000, 
        '2.features_size' : [10000], #500, 1000, 5000, 10000, 50000, 90000], #[500, 1000, 5000, 10000, 50000, 90000],
        #'3.Bagging' : [0.66],
        '4.Iterations' : [100], #[10, 50, 100],
        '3.Sigma' : [0.5] #, 0.7, 0.9]
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
    ax = fig.add_subplot(111)

    for i, params in enumerate(grid_search.IterGrid(params_range)):

        if params['1.vocab_size'] > params['2.features_size']:

            params_path = "/".join( [ key.split('.')[1] + str(value).replace('.','') for key, value in sorted( params.items() ) ] )
            params_path = '/' + params_path
            
            #color = color_pallet[ params['1.vocab_size'] ]
            color = color_pallet2[ params['2.features_size'] ]

            fidx2gnr = Idx_2_Genre(root_path_Char4G, 'Corpus_filename_shorted.lst')
            X_sets, Y_sets, Z_sets = Docs_Sizes(res_h5file, kfolds, params_path, root_path_Char4G)

            if Z_sets[0]:
                ax.projection('3d')
                
            for zi, (x, y, z) in enumerate(zip(X_sets, Y_sets, Z_sets)):
                
                hist, bins = np.histogram(y, 200)
                print hist, bins
                tmp = np.zeros(len(bins+1))
                tmp[0:-1] = hist[::]
                hist = tmp

                if Z_sets[0]:
                    ax.bar(bins, hist, zi, zdir='y', alpha=0.8, align='center', width=bins[1]-bins[0])
        
                else:
                    ax.bar(bins, hist, alpha=0.8, align='center', width=bins[1]-bins[0])
            
            #ax.set_ylim3d(0,11) 
            #ax.set_xlim3d(0,60000) 
            if Z_sets[0]:
                plt.yticks(range(12), Z_sets, size="small")

            #plt.title('False Negative("Don''t Know" excluded) Document-Size-Distributions per Genre (Char4Grams)')
            plt.title('"Don''t Know" Document-Size-Distributions per Genre (Char4Grams)')
            #ax.w_yaxis.set_ticklabels(Z_sets, size="small") 
            
            
            #sub1.legend(loc=3, bbox_to_anchor=(0.08, -0.4), ncol=2, fancybox=True, shadow=True)    
           
    ax.xaxis.set_ticks(np.arange(0, 300010, 10000))
    ax.yaxis.set_ticks(np.arange(0, 100, 5))
    plt.setp(plt.xticks()[1], rotation=45, size='small')
    if Z_sets[0]:
        plt.setp(plt.zticks()[1], rotation=45, size='small')

    #plt.savefig('/home/dimitrios/Desktop/Expected_ZClass.pdf')
    ax.set_xlabel('# Terms')
    
    if Z_sets[0]:
        ax.set_zlabel('# Documents')
        ax.set_ylabel('Genres')
    else:
        ax.set_ylabel('# Documents')

    plt.show()
                                                                             
    res_h5file.close()   