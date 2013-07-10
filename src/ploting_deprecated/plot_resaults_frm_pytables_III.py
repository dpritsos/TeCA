
" Precision vs. Recall Diagrams & F3 statistic"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt


def plot_data(Res, kfolds, featr_size_lst, genres):
      
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
    symbol = [ "*", "^", "x", "+", "^" , "*", "+", "x" ]
    line_type = [ "-", "-", "-", "-", "--" , "--", "--", "--" ]
    
    F05_per_feat_size = list()
    P_per_feat_size = list()
    R_per_feat_size = list()
        
    for featr_size in featr_size_lst:
        
        kfold_F05 = list()
        kfold_P = list()
        kfold_R = list()
        
        for k in range(kfolds):
            
            kfold_F05.append( Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100'+'/Sigma0.9', name='F05_per_gnr' ) ) #+'/Sigma0.9'
            kfold_P.append( Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100'+'/Sigma0.9', name='P_per_gnr' ) )
            kfold_R.append( Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100'+'/Sigma0.9', name='R_per_gnr' ) )
        
        F05_per_feat_size.append( np.mean(np.vstack(kfold_F05), axis=0) )
        P_per_feat_size.append( np.mean(np.vstack(kfold_P), axis=0) )
        R_per_feat_size.append( np.mean(np.vstack(kfold_R), axis=0) )
        
    P = np.vstack(P_per_feat_size)
    R = np.vstack(R_per_feat_size)
    F05 = np.vstack(F05_per_feat_size)
    
    for row in P[:, 1::]:
        print "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % tuple(row)
    #print P[:, 1::],"\n"
    #print R[:, 1::],"\n"
    #Remove Zero Counts for Uncategorised Genres 
    #F05 = F05[1::, :]
    #P = P[1::, :]
    #R = R[1::, :]
    
    #Start Ploting figures
    #plt.figure(0)
    #plt.title( "Corpus: Santini's | TermsType: Words | Text Modeling: TF | Koppels Method " )
    
    #Plot all F1 Scores for all genre and all features sizes in one plot 
    #plt.subplot(3,1, 1)
    plt.figure(0)
    #plt.title( "Corpus: Santini's | TermsType: Words | Text Modeling: BIN | Koppel's" )
    plt.xlabel( 'Features' )
    plt.ylabel( 'F1' ) 
    for g in range(F1.shape[0]):
        plt.plot(np.array(featr_size_lst)[:],\
                 F1[g,:], color[g] + symbol[g] + line_type[g],\
                 label=genres[g])
        plt.legend(loc=4 )
    
    plt.grid(True)
        
    #Plot all P Scores for all genre and all features sizes in one plot 
    #plt.subplot(3,1, 2)
    plt.figure(1)
    #plt.title( "Corpus: Santini's | TermsType: Words | Text Modeling: BIN | Koppel's" )
    plt.xlabel( 'Features' )
    plt.ylabel( 'P' ) 
    for g in range(P.shape[0]):
        plt.plot(np.array(featr_size_lst)[:],\
                 P[g,:], color[g] + symbol[g] + line_type[g],\
                 label=genres[g])
        plt.legend(loc=4 )
    
    plt.grid(True)
    
    #Plot all R Scores for all genre and all features sizes in one plot 
    #plt.subplot(3,1, 3)
    plt.figure(2)
    #plt.title( "Corpus: Santini's | TermsType: Words | Text Modeling: BIN | Koppel's" )
    plt.xlabel( 'Features' )
    plt.ylabel( 'R' ) 
    for g in range(R.shape[0]):
        plt.plot(np.array(featr_size_lst)[:],\
                 R[g,:], color[g] + symbol[g] + line_type[g],\
                 label=genres[g])
        plt.legend(loc=4 )
            
    plt.grid(True)
    plt.Figure()
    plt.show()
                 

if __name__ == '__main__':
    
    genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop" ]
    #genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    kfolds = 10
    featr_size_lst = [1000, 5000, 10000, 20000, 50000, 70000]
    gnr_num = 7
    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Matthews_correlation.h5', 'r')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Char4Grams-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'r')
    
                                                                                    
    plot_data(CrossVal_Kopples_method_res, kfolds, featr_size_lst, genres)
    
    CrossVal_Kopples_method_res.close()
    
