
" Precision vs. Recall Diagrams & F3 statistic"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import PR_curves_to_11_standard_recall_levels as srl

def plot_data(Res, kfolds, featr_size_lst, genres):
      
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
    symbol = [ "*", "^", "x", "+", "^" , "*", "+", "x" ]
    line_type = [ "-", "-", "-", "-", "--" , "--", "--", "--" ]
    
    fig = 0
    
    for featr_size in featr_size_lst:
        
        #predicted_scores = list()
        #expected_Y = list()
        
        plt.figure(num=fig, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
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

            inv_srd_idxs = np.argsort(PS)[::-1]
            PS = PS[ inv_srd_idxs ]
            EY = EY[ inv_srd_idxs ]
            
            P, R, T= skm.precision_recall_curve(EY, PS)
            
            x, y = srl.STD_AVG_PR(P, R)
            
            #Plot all F1 Scores for all genre and all features sizes in one plot 
            #plt.subplot(3,1, 1)
            
            plt.title('(a)')
            plt.xlabel( 'R' )
            plt.ylabel( 'P' ) 
            plt.plot(x, y, color[g_num] + symbol[g_num] + line_type[g_num], label=genres[g_num])
            plt.grid(True)
            plt.legend(loc=3)
    
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #y = [1.0, 1.0, 0.97190657, 0.92278167, 0.88407108, 0.86455453, 0.85036651, 0.84308633, 0.83109285, 0.81946031, 0.79872149] #Santinis - C4Grams
    y = [1.0, 1.0, 0.97379638, 0.94235053, 0.89743401, 0.87974863, 0.86553602, 0.85686608, 0.84702267, 0.83625077, 0.81514552]
    #y = [ 1., 1., 0.97591298, 0.91989472, 0.85962455, 0.82631012, 0.7482368,  0.72640206, 0.70890553, 0.66452836, 0.5594028 ] #KI-04 Char4Grams
    plt.plot(x, y, 'k-', linewidth=2, label=genres[g_num])
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
    
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Char4Grams-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    
                                                                                    
    plot_data(CrossVal_Kopples_method_res, kfolds, featr_size_lst, genres)
    
    CrossVal_Kopples_method_res.close()
    
