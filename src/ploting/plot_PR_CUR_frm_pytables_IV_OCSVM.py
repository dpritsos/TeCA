
" Precision vs. Recall Diagrams & F3 statistic"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm

def plot_data(Res, kfolds, featr_size_lst, nu_lst):
      
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'  ]
    symbol = [ "*", "^", "x", "+", "^" , "*", "+", "x", "*", "^", "x", "+", "^" , "*", "+", "x" ]
    line_type = [ "-", "-", "-", "-", "--" , "--", "--", "--", "-", "-", "-", "-", "--" , "--", "--", "--" ]
    
    plt.figure(0)
    
    
    for i_fs, featr_size in enumerate(featr_size_lst):
        
        for i_nu, nu in enumerate(nu_lst):
            
            DS_lst = list()
            PY_lst = list()
                
            for k in range(kfolds):     
                
                ds_per_gnr = Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='predicted_Dist_per_gnr').read()
                DS_lst.append( np.max(ds_per_gnr, axis=0)[0:-1:10] )
                p_y = np.argmax(ds_per_gnr, axis=0)+1 
                exp_y = Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='expected_Y' ).read()
                PY_lst.append( np.where(p_y[0:-1:10] == exp_y[0:-1:10], 1, 0) )
                #pre_y_per_gnr = Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='predicted_Y_per_gnr' ).read()
                #print pre_y_per_gnr
                #print np.argmax(pre_y_per_gnr, axis=0)+1
                #print p_y
            
            DS = np.hstack(DS_lst)
            PY = np.hstack(PY_lst)
                    
            inv_srd_idxs = np.argsort(DS)[::-1]
            DS = DS[ inv_srd_idxs ]
            PY = PY[ inv_srd_idxs ]
            
            P, R, T= skm.precision_recall_curve(PY, DS)
            
            #Plot all F1 Scores for all genre and all features sizes in one plot 
            #plt.subplot(3,1, 1)
            
            #plt.title( "Corpus: Santini's | TermsType: Words | Text Modeling: BIN | Koppel's" )
            plt.xlabel( 'R' )
            plt.ylabel( 'P' ) 
            plt.plot(R, P, color[i_fs] + symbol[i_fs] + line_type[i_fs], label=str(featr_size)+" - "+str(nu))
            plt.grid(True)    
            plt.legend(loc=3 )
                    
    plt.Figure()
    plt.show()
        
    
                 

if __name__ == '__main__':
    
    kfolds = 10
    #featr_size_lst = [1000, 5000, 10000, 20000, 50000, 70000]
    featr_size_lst = [1000, 5000, 10000, 70000]
    gnr_num = 7
    nu_lst = [0.8] #[0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8]
    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-OC-SVM_kfolds-10_TM-TF_(DIST).h5', 'r')
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Char4Grams-OC-SVM_kfolds-10_TM-TF_(DIST).h5', 'r')  
                                                                                    
    plot_data(CrossVal_Kopples_method_res, kfolds, featr_size_lst, nu_lst)
    
    CrossVal_Kopples_method_res.close()
    
