

import tables as tb
import numpy as np
import matplotlib.pyplot as plt


def plot_data(Res, kfolds, featr_size_lst, genres):
      
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
    symbol = [ "*", "^", "x", "+", "^" , "*", "+", "x" ]
    line_type = [ "-", "-", "-", "-", "--" , "--", "--", "--" ]
    ylbl = ['F1', 'P', 'R']
    
    F1_per_feat_size = list()
    P_per_feat_size = list()
    R_per_feat_size = list()
        
    for featr_size in featr_size_lst:
        
        kfold_F1 = list()
        kfold_P = list()
        kfold_R = list()
        
        for k in range(kfolds):
            
            kfold_F1.append( Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='F1_per_gnr' ) )
            kfold_P.append( Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='P_per_gnr' ) )
            kfold_R.append( Res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters100', name='R_per_gnr' ) )
        
        F1_per_feat_size.append( np.mean(np.vstack(kfold_F1), axis=0) )
        P_per_feat_size.append( np.mean(np.vstack(kfold_P), axis=0) )
        R_per_feat_size.append( np.mean(np.vstack(kfold_R), axis=0) )
    
    print F1_per_feat_size
    print P_per_feat_size
    print R_per_feat_size
        
    F1 = np.vstack(F1_per_feat_size).T
    P = np.vstack(P_per_feat_size).T
    R = np.vstack(R_per_feat_size).T
    
    #Remove Zero Counts for Uncategorised Genres 
    F1 = F1[1::, :]
    P = P[1::, :]
    R = R[1::, :]
    
    #Start Ploting figures
    #plt.figure(0)
    #plt.title( "Corpus: Santini's | TermsType: Words | Text Modeling: TF | Koppels Method " )
    
    #Plot all F1 Scores for all genre and all features sizes in one plot 
    #plt.subplot(3,1, 1)
    plt.figure(0)
    plt.title( "Corpus: KI-04 | TermsType: Words | Text Modeling: TF | Koppel's Method " )
    
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
    plt.title( "Corpus: KI-04 | TermsType: Words | Text Modeling: TF | Koppel's Method " )
    
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
    plt.title( "Corpus: KI-04 | TermsType: Words | Text Modeling: TF | Koppel's Method " )
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
    
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'r')    
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Char4Grams-Koppels_method_kfolds-10_SigmaThreshold-None_nrmMAX.h5', 'r')
    
    plot_data(CrossVal_Kopples_method_res, kfolds, featr_size_lst, genres)
    
    CrossVal_Kopples_method_res.close()
    
