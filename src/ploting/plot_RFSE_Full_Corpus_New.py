
" Precision vs. Recall Diagrams & F3 statistic"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn import grid_search
import PR_curves_to_11_standard_recall_levels as srl


def plot_data(Res, params_range):
      
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
    symbol = [ "^", "*", "x", "+", "*", "^", "x", "+" ]
    line_type = [ "--", "--", "--", "--", "-" , "-", "-", "-" ]
    
    for params in grid_search.IterGrid(params_range):

        plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

        ZL_lst = list()

        for i_fs, featr_size in enumerate(params['features_size']):

            if params['vocab_size'] < featr_size:
                continue
                   
            PS_lst = list() #Ensemble Algo Predicted Scores
            TT_lst = list() #Ensemble Algo Truth Table
            full_PY_lst = list()
                    
            for k in range(params['kfolds']):  
                
                h5_node_path = '/Vocab' + str( params['vocab_size'] )
                h5_node_path += '/Feat' + str( featr_size )
                h5_node_path += '/Iters' + str( params['training_iter'] )
                h5_node_path += '/Sigma' + str( params['threshold'] )
                h5_node_path += '/Bagg' + str( params['bagging_param'] )
                h5_node_path += '/KFold' + str( k )

                plot_title = '/Vocab' + str( params['vocab_size'] )
                plot_title += '/Iters' + str( params['training_iter'] )
                plot_title += '/Sigma' + str( params['threshold'] )
                plot_title += '/Bagg' + str( params['bagging_param'] )

                PS_lst.append( Res.getNode(h5_node_path, name='predicted_scores' ).read() )
                
                exp_y = Res.getNode(h5_node_path, name='expected_Y' ).read()
                pre_y = Res.getNode(h5_node_path, name='predicted_Y' ).read()

                TT_lst.append( np.where(exp_y == pre_y, 1, 0) )

                full_PY_lst.append(pre_y)

            #Make Tables for Ensemble Algo
            PS = np.hstack(PS_lst)
            TT = np.hstack(TT_lst)
            FPY = np.hstack(full_PY_lst)
            
            #Short Results by Predicted Scores
            inv_srd_idx = np.argsort(PS)[::-1]
            PS = PS[ inv_srd_idx ]
            TT = TT[ inv_srd_idx ]
            FPY = FPY[ inv_srd_idx ]
            ZL_lst.append( np.where(FPY == 0) )
                           
            #Calculate P-R Curves for Ensemble Algorithm
            P, R, T= skm.precision_recall_curve(TT, PS)
            
            #print T
            T = T[::-1]
            R = R[::-1]
            P = P[::-1]
            a = np.max(np.where(T >= 0.8))
            #print T
            #print R
            #print P
            #print np.where(PS >= 0.5)
            
            plt.subplot(2, 1, 1)

            plt.plot(R[a], P[a], color[i_fs] + symbol[i_fs], markeredgewidth=15)
            
            x, y = srl.STD_AVG_PR(P[::-1], R[::-1])
            
            x = R
            y, x = srl.INTERPOL_SMOOTH_PR(P[::-1], R)
            y = y
            
            plt.plot(x, y, color[i_fs] + symbol[i_fs] + line_type[i_fs], markeredgewidth=2, label="RFSE. - "+str(featr_size) )
            
            plt.title(plot_title)
            plt.xlabel('R')
            plt.ylabel('P')  
            
            plt.grid(True)    
            plt.legend(loc=3)

        plt.subplot(2, 1, 2)
        plt.boxplot(ZL_lst)
                              
    plt.Figure()
    plt.show()
        
                 

if __name__ == '__main__':

    params_range = {
        'kfolds' : [10],
        'vocab_size' : [10000, 50000, 100000],
        'features_size' : [[1000, 5000, 10000, 70000]],
        'training_iter' : [100],
        'threshold' : [0.5, 0.8],
        'bagging_param' : [0.33, 0.66],
    } 
         
    EnsAlgo = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Char4Grams-Koppels-Bagging_method_kfolds-10_GridSearch_TEST.h5', 'r')
                                                                         
    plot_data(EnsAlgo, params_range)
    
    EnsAlgo.close()
    
