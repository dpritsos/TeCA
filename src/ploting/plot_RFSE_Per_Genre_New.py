
" Precision vs. Recall Diagrams & F3 statistic"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn import grid_search
import PR_curves_to_11_standard_recall_levels as srl

def plot_data(Res, params_range):
      
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
    symbol = [ "*", "^", "x", "+", "^" , "*", "+", "x" ]
    line_type = [ "-", "-", "-", "-", "--" , "--", "--", "--" ]
    
    for params in grid_search.IterGrid(params_range):

        for i_fs, featr_size in enumerate(params['features_size']):

            if params['vocab_size'] < featr_size :
                continue

            plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

            ZL_lst = list()

            for g_num in range(len(params['genres'])):
                
                g_tag = g_num + 1
                
                ps_per_fold = list()
                ey_per_fold = list()
                pre_y_per_gnr = list()
                
                for k in range(params['kfolds']):

                    h5_node_path = '/Vocab' + str( params['vocab_size'] )
                    h5_node_path += '/Feat' + str( featr_size )
                    h5_node_path += '/Iters' + str( params['training_iter'] )
                    h5_node_path += '/Sigma' + str( params['threshold'] )
                    h5_node_path += '/Bagg' + str( params['bagging_param'] )
                    h5_node_path += '/KFold' + str( k )

                    plot_title = '/Vocab' + str( params['vocab_size'] )
                    plot_title += '/Feat' + str( featr_size )
                    plot_title += '/Iters' + str( params['training_iter'] )
                    plot_title += '/Sigma' + str( params['threshold'] )
                    plot_title += '/Bagg' + str( params['bagging_param'] )
                    
                    pc_per_iter = Res.getNode(h5_node_path, name='predicted_classes_per_iter').read()
                    gnr_pred_cnt = np.where(pc_per_iter == g_tag, 1, 0) 
                    fold_ps = np.sum(gnr_pred_cnt, axis=0) / np.float(pc_per_iter.shape[0])   
                    ps_per_fold.append(fold_ps) 
                    
                    exp_y = Res.getNode(h5_node_path, name='expected_Y' ).read()
                    ey_per_fold.append( np.where(exp_y == g_tag, 1, 0) ) #Covert exp_y to Binary case and append for this fold
                    
                    pre_y = Res.getNode(h5_node_path, name='predicted_Y' ).read()
                    pre_y_per_gnr.append( pre_y )
                
                PS = np.hstack(ps_per_fold)
                EY = np.hstack(ey_per_fold)
                PY_per_gnr = np.hstack(pre_y_per_gnr)

                inv_srd_idxs = np.argsort(PS)[::-1]
                PS = PS[ inv_srd_idxs ]
                EY = EY[ inv_srd_idxs ]
                PY_per_gnr = PY_per_gnr[ inv_srd_idxs ]
                ZL_lst.append( np.where(PY_per_gnr == 0) )

                P, R, T= skm.precision_recall_curve(EY, PS)
                
                x, y = srl.STD_AVG_PR(P, R)
                        
                plt.subplot(2, 1, 1)
                plt.ylim([0.4,1.0])
                plt.plot(x, y, color[g_num] + symbol[g_num] + line_type[g_num], markeredgewidth=2, label=params['genres'][g_num])
                plt.title(plot_title)

                plt.xlabel( 'R' )
                plt.ylabel( 'P' ) 
                
                plt.grid(True)
                plt.legend(loc=3)
            
            plt.subplot(2,1, 2)
            plt.boxplot(ZL_lst)

    plt.Figure()
    plt.show()
        
    
                 

if __name__ == '__main__':

    params_range = {
        'kfolds' : [10],
        'vocab_size' : [10000, 100000], #[10000, 50000, 100000],
        'features_size' : [[1000, 5000, 10000, 70000]],
        'training_iter' : [100],
        'threshold' : [0.5], #, 0.8],
        'bagging_param' : [0.66], #, 0.33],
        'genres' : [["article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop"]],
        #'genres' : [["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]]
    } 
        
    #

    EnsAlgo = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Char4Grams-Koppels-Bagging_method_kfolds-10_GridSearch_TEST.h5', 'r')    
    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Char4Grams-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Char4Grams-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Char4Grams-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
                                                                             
    plot_data(EnsAlgo, params_range)
    
    EnsAlgo.close()
    
