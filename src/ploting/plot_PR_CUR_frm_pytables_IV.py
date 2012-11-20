
" Precision vs. Recall Diagrams & F3 statistic"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm

def plot_data(Res, kfolds, featr_size_lst):
      
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
    symbol = [ "*", "^", "x", "+", "^" , "*", "+", "x" ]
    line_type = [ "-", "-", "-", "-", "--" , "--", "--", "--" ]
    
    plt.figure(0)
    
    for i_fs, featr_size in enumerate(featr_size_lst):
        
        ps = list()
        trueth_tbl = list()
                
        for k in range(kfolds):     
            
            ps.append( Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='predicted_scores' ).read() )
            exp_y = Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='expected_Y' ).read()
            pre_y = Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='predicted_Y' ).read()
            trueth_tbl.append( np.where(exp_y == pre_y, 1, 0) ) #Covert exp_y to Binary case and append for this fold
                
        PS = np.hstack(ps)
        TT = np.hstack(trueth_tbl)
        
        inv_srd_idxs = np.argsort(PS)[::-1]
        PS = PS[ inv_srd_idxs ]
        TT = TT[ inv_srd_idxs ]      
    
        P, R, T= skm.precision_recall_curve(TT, PS)
        
        #plt.title( "Corpus: Santini's | TermsType: Words | Text Modeling: BIN | Koppel's" )
        plt.xlabel( 'R' )
        plt.ylabel( 'P' ) 
        plt.plot(R, P, color[i_fs] + symbol[i_fs] + line_type[i_fs], label=str(featr_size))
        plt.grid(True)    
    
    plt.legend(loc=3 )        
    plt.Figure()
    plt.show()
        
    
                 

if __name__ == '__main__':
    
    kfolds = 10
    #featr_size_lst = [1000, 5000, 10000, 20000, 50000, 70000]
    featr_size_lst = [1000, 5000, 10000, 70000]
    gnr_num = 7
    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Char4Grams-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Char4Grams-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    
                                                                                    
    plot_data(CrossVal_Kopples_method_res, kfolds, featr_size_lst)
    
    CrossVal_Kopples_method_res.close()
    
