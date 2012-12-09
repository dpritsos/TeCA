
" Precision vs. Recall Diagrams & F3 statistic"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import PR_curves_to_11_standard_recall_levels as srl


def plot_data(Res, kfolds, featr_size_lst, genres, nu):
      
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
            
            ds_per_fold = list()
            ey_per_fold = list()
            
            for k in range(kfolds):     
                
                ds_per_gnr = Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='predicted_Dist_per_gnr').read()   
                ds_per_fold.append( ds_per_gnr[g_num][::10] ) 
                exp_y = Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='expected_Y' ).read()
                ey_per_fold.append( np.where(exp_y[::10] == g_tag, 1, 0) ) #Covert exp_y to Binary case and append for this fold
                
            DS = np.hstack(ds_per_fold)
            EY = np.hstack(ey_per_fold)
            
            inv_srd_idxs = np.argsort(DS)[::-1]
            DS = DS[ inv_srd_idxs ]
            EY = EY[ inv_srd_idxs ]
            
            P, R, T= skm.precision_recall_curve(EY, DS)
            
            x, y = srl.STD_AVG_PR(P, R)
            
            #Plot all F1 Scores for all genre and all features sizes in one plot 
            #plt.subplot(3,1, 1)
            
            plt.title('(b)')
            plt.xlabel( 'R' )
            plt.ylabel( 'P' ) 
            plt.plot(x, y, color[g_num] + symbol[g_num] + line_type[g_num], label=genres[g_num])
            plt.grid(True)
            plt.legend(loc=1)
        
    plt.Figure()
    plt.show()
        
    
                 

if __name__ == '__main__':
    
    genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop" ]
    #genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    kfolds = 10
    #featr_size_lst = [1000, 5000, 10000, 20000, 50000, 70000]
    featr_size_lst = [5000]
    gnr_num = 7
    nu = 0.8
    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Char4Grams-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Char4Grams-OC-SVM_kfolds-10_TM-TF_(DIST).h5', 'r')
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Char4Grams-OC-SVM_kfolds-10_TM-TF_(DIST).h5', 'r')
    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Char4Grams-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    
                                                                                    
    plot_data(CrossVal_Kopples_method_res, kfolds, featr_size_lst, genres, nu)
    
    CrossVal_Kopples_method_res.close()
    
