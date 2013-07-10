
" Precision vs. Recall Diagrams & F3 statistic"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt


def plot_data(Res, kfolds, featr_size_lst, nu_lst, genres):
      
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
    symbol = [ "*", "^", "x", "+", "^" , "*", "+", "x" ]
    line_type = [ "-", "-", "-", "-", "--" , "--", "--", "--" ]
    
    
    fg_cnt = 0    
    for featr_size in featr_size_lst:
        
        F05_per_nu = list()
        P_per_nu = list()
        R_per_nu = list()
        
        for nu in nu_lst:
        
            kfold_F05 = list()
            kfold_P = list()
            kfold_R = list()
            
            for k in range(kfolds):
                
                kfold_F05.append( Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='F05_per_gnr' ) ) #+'/Sigma0.9'
                kfold_P.append( Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='P_per_gnr' ) )
                kfold_R.append( Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='R_per_gnr' ) )
            
            F05_per_nu.append( np.mean(np.vstack(kfold_F05), axis=0) )
            P_per_nu.append( np.mean(np.vstack(kfold_P), axis=0) )
            R_per_nu.append( np.mean(np.vstack(kfold_R), axis=0) )
            
        P = np.vstack(P_per_nu)
        R = np.vstack(R_per_nu)
        F05 = np.vstack(F05_per_nu)
        
        #for row in P[:, 1::]:
        #    print "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % tuple(row)
         
        PT = P.T[1::] 
        RT = R.T[1::]
        
        plt.figure(fg_cnt)
        fg_cnt += 1
        #plt.title( "Corpus: Santini's | TermsType: Words | Text Modeling: BIN | Koppel's" )
        plt.xlabel( 'Features' )
        plt.ylabel( 'F1' ) 
        
        print str(fg_cnt)+". R=",np.mean(RT, axis=0)
        print str(fg_cnt)+". P=",np.mean(PT, axis=0)
        
        plt.plot(np.array(nu_lst),\
                 np.mean(PT, axis=0),\
                 color[1] + symbol[1] + line_type[1],\
                 label="CURVE")
        plt.legend(loc=4 )
        
        plt.plot(np.array(nu_lst),\
                 np.mean(RT, axis=0),\
                 color[2] + symbol[2] + line_type[2],\
                 label="CURVE")
        plt.legend(loc=4 )
        
        plt.plot(np.mean(RT[::-1], axis=0),\
                 np.mean(PT, axis=0),\
                 color='grey', lw=3,\
                 label="CURVE")
        plt.legend(loc=4 )
        
        plt.grid(True)
            
       
    plt.Figure()
    plt.show()
                 

if __name__ == '__main__':
    
    #genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop" ]
    genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    kfolds = 10
    nu_l = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8]
    featr_size_lst = [1000, 5000, 10000, 20000, 50000, 70000]
    gnr_num = 7
    
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Char4Grams-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Char4Grams-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    
                                                                                    
    plot_data(CrossVal_Kopples_method_res, kfolds, featr_size_lst, nu_l, genres)
    
    CrossVal_Kopples_method_res.close()
    
