
" Precision vs. Recall Diagrams & F3 statistic"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import PR_curves_to_11_standard_recall_levels as srl


def plot_data(Res1, Res2, kfolds, featr_size_lst1, featr_size_lst2, nu):
      
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
    symbol = [ "^", "*", "x", "+", "*", "^", "x", "+" ]
    line_type = [ "--", "--", "--", "--", "-" , "-", "-", "-" ]
    
    plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    
    for i_fs, featr_size in enumerate(featr_size_lst1):
          
        DS_lst = list() #Ensemble OC-SVM Distance Scores
        PY_lst = list() #Ensemble OC-SVM Predicted Ys           
        PS_lst = list() #Ensemble Algo Predicted Scores
        TT_lst = list() #Ensemble Algo Truth Table
                
        for k in range(kfolds):  
            
            if featr_size in featr_size_lst2:
                ds_per_gnr = Res1.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='predicted_Dist_per_gnr').read()
                rpy_per_gnr = Res1.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='predicted_Y_per_gnr').read()
                RPY = np.max(rpy_per_gnr, axis=0)[::]  
                DS_lst.append( np.max(ds_per_gnr, axis=0)[::] )            
                p_y = np.argmax(ds_per_gnr, axis=0)+1
                #print p_y
                p_y[ np.where(RPY < 0) ] = 0
                #print p_y 
                exp_y = Res1.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='expected_Y' ).read()
                PY_lst.append( np.where(p_y[::] == exp_y[::], 1, 0) )
            
            PS_lst.append( Res2.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='predicted_scores' ).read() )
            exp_y = Res2.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='expected_Y' ).read()
            pre_y = Res2.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='predicted_Y' ).read()
            
            TT_lst.append( np.where(exp_y == pre_y, 1, 0) ) #Covert exp_y to Binary case and append for this fold
            
        
        #Make Tables for Ensemble OC-SVM
        if featr_size in featr_size_lst2:
            DS = np.hstack(DS_lst) 
            PY = np.hstack(PY_lst)
        #Make Tables for Ensemble Algo
        PS = np.hstack(PS_lst)
        TT = np.hstack(TT_lst)
        
        #Short Results by Distance Scores
        if featr_size in featr_size_lst2:
            inv_srd_idxs = np.argsort(DS)[::-1]
            DS = DS[ inv_srd_idxs ]
            PY = PY[ inv_srd_idxs ]
        #Short Results by Predicted Scores
        inv_srd_idx = np.argsort(PS)[::-1]
        PS = PS[ inv_srd_idx ]
        TT = TT[ inv_srd_idx ]
        
        #print PS
        
        """
        #Calculate P-R Curves for Ensemble OC-SVM
        P, R, T= skm.precision_recall_curve(PY, DS)
        
        x, y = srl.STD_AVG_PR(P, R)
        
        print y
        
        plt.plot(x, y, color[i_fs] + symbol[i_fs] + line_type[i_fs], label="O.S.E. - "+str(featr_size) )
        """
                
        #Calculate P-R Curves for Ensemble Algorithm
        P, R, T= skm.precision_recall_curve(TT, PS)
        
        #print T
        T = T[::-1]
        R = R[::-1]
        P = P[::-1]
        a = np.max(np.where(T >= 0.9))
        print a
        #print T
        #print R
        #print P
        #print np.where(PS >= 0.5)
        
        plt.plot(R[a], P[a], color[i_fs] + symbol[i_fs], markeredgewidth=15)
        

        x, y = srl.STD_AVG_PR(P[::-1], R[::-1])
        
        x = R
        y, x = srl.INTERPOL_SMOOTH_PR(P[::-1], R)
        y = y
        #plt.title( ) 
        plt.plot(x, y, color[i_fs] + symbol[i_fs] + line_type[i_fs], markeredgewidth=2, label="RFSE. - "+str(featr_size) )
        
        #plt.title('(a)')
        plt.xlabel('R')
        plt.ylabel('P')  
    

    P, R, T= skm.precision_recall_curve(PY, DS)
        
    #x, y = srl.STD_AVG_PR(P, R)
    
   
    
    R = R[::-1]
    
    
    y, x = srl.INTERPOL_SMOOTH_PR(P, R)
       
    plt.plot(x, y, color[i_fs+4] + line_type[i_fs+4], linewidth=2, label="BASELINE" )  
    
    plt.grid(True)    
    plt.legend(loc=3)
                              
    plt.Figure()
    plt.show()
        
                 

if __name__ == '__main__':
    
    kfolds = 10
    featr_size_lst1 = [1000, 5000, 10000, 70000] #[1000, 5000, 10000, 20000, 50000, 70000]
    featr_size_lst2 = [5000]
    gnr_num = 7
    nu = 0.1#[0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8]
    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-OC-SVM_kfolds-10_TM-TF_(DIST).h5', 'r')

    EnsOCSVM = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Words-OC-SVM_kfolds-10_TM-TF_(DIST).h5', 'r')
    EnsAlgo = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')  
    
    #EnsOCSVM = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-OC-SVM_kfolds-10_TM-TF_(DIST).h5', 'r')
    #EnsAlgo = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r') 

                                                                                    
    plot_data(EnsOCSVM, EnsAlgo, kfolds, featr_size_lst1, featr_size_lst2, nu)
    
    EnsOCSVM.close()
    EnsAlgo.close()
    
