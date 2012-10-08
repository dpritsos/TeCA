
" Precision vs. Recall Diagrams & F3 statistic"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm

def plot_data(Res, kfolds, featr_size_lst, genres):
      
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
    symbol = [ "*", "^", "x", "+", "^" , "*", "+", "x" ]
    line_type = [ "-", "-", "-", "-", "--" , "--", "--", "--" ]
    
    fig = 0
    
    for featr_size in featr_size_lst:
        
        #predicted_scores = list()
        #expected_Y = list()
        
        plt.figure(fig)
        fig += 1
        
        for g_num in range(len(genres)):
            
            g_tag = g_num + 1
            
            ps_per_fold = list()
            ey_per_fold = list()
            
            for k in range(kfolds):     
                
                pc_per_iter = Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='predicted_classes_per_iter').read()
                gnr_pred_cnt = np.where(pc_per_iter == g_tag, 1, 0) 
                fold_ps = np.sum(gnr_pred_cnt, axis=0)/np.float(pc_per_iter.shape[0])   
                
                ps_per_fold.append(fold_ps) #Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='predicted_scores' ).read() )
                exp_y = Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='expected_Y' ).read()
                ey_per_fold.append( np.where(exp_y == g_tag, 1, 0) ) #Covert exp_y to Binary case and append for this fold
                
            PS = np.hstack(ps_per_fold)
            EY = np.hstack(ey_per_fold)
            
            inv_srd_idxs = np.argsort(PS)[::1]
            PS = PS[ inv_srd_idxs ]
            EY = EY[ inv_srd_idxs ]
            
            P, R, T= skm.precision_recall_curve(EY, PS)
            
            
            #Plot all F1 Scores for all genre and all features sizes in one plot 
            #plt.subplot(3,1, 1)
            
            #plt.title( "Corpus: Santini's | TermsType: Words | Text Modeling: BIN | Koppel's" )
            plt.xlabel( 'R' )
            plt.ylabel( 'P' ) 
            plt.plot(R, P, color[g_num] + symbol[g_num] + line_type[g_num], label=genres[g_num])
            plt.grid(True)
                
        plt.plot([0.0,1.0],[0.488, 0.488], 'k-', label='Baseline')
        plt.legend(loc=3 ) 
    plt.Figure()
    plt.show()
        
    
                 

if __name__ == '__main__':
    
    genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop" ]
    #genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    kfolds = 10
    #featr_size_lst = [1000, 5000, 10000, 20000, 50000, 70000]
    featr_size_lst = [5000]
    gnr_num = 7
    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Char4Grams-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Char4Grams-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    
                                                                                    
    plot_data(CrossVal_Kopples_method_res, kfolds, featr_size_lst, genres)
    
    CrossVal_Kopples_method_res.close()
    
