

import tables as tb
import numpy as np
import matplotlib.pyplot as plt


class MultiResultsTable_desc(tb.IsDescription):
    kfold = tb.UInt32Col(pos=1)
    Acc = tb.Float32Col(pos=2)
    feat_num_rq = tb.UInt32Col(pos=3)
    feat_num_rt = tb.UInt32Col(pos=4)
    Centroids_num = tb.UInt32Col(pos=5)
    Centroids = tb.Float32Col(shape=(1,10), pos=6)
    Sigma = tb.Float32Col(pos=7)
    Predicted_Y = tb.Float32Col(shape=(1,5000),pos=8)
    Expected_Y = tb.Float32Col(shape=(1,5000),pos=9)
    SVM_C = tb.Float32Col(pos=10)



def plot_data(res_tb, sv_tb,  featr_size_lst, centroids_ll, sigma_l):
    #print res_tb.read()
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
    symbol = [ "*", "^", "x", "x", "+" , "+", "*", "x", "+" ]
    line_type = [ "-", "-", "-", "-", "-" , "--", "--", "--", "--" ]
    
    ylbl = ['F1', 'Prec', 'Recall', "Accuracy"]
    plt.figure( 1 )
    plt_pos = 0 
    plt.title( "Corpus: KI-04 | TermsType: Words | Text Modelling: Derivative(>0)" )
    
    
    for centroids_l in centroids_ll[:]: 
        centrds_num = len(centroids_l)
        tplt_l = list()
        for featr_size in featr_size_lst:   
            #centrd = np.zeros((1,10))
            #centrd[0,np.arange(len(centroids_l))] = np.array(centroids_l)
            #print res_tb.read()
            for sigma in sigma_l:
                #print '(feat_num_rq == ' +str(featr_size)+ 'L) & (Centroids_num == ' +str(centrds_num)+ 'L) & (Sigma >= ' +str(np.float32(sigma))+ ' )'
                kfolds_acc = [ row['Acc'] for row in\
                    res_tb.where('(feat_num_rq == ' +str(featr_size)+')')] #'+ 'L) & (Centroids_num == ' +str(centrds_num)+ 'L) & (Sigma >= ' +str(np.float32(sigma))+ ' )')]
                kfolds_feat_num = [ row['feat_num_rt'] for row in\
                    res_tb.where('(feat_num_rq == ' +str(featr_size)+')')] #+ 'L) & (Centroids_num == ' +str(centrds_num)+ 'L) & (Sigma >= ' +str(np.float32(sigma))+ ' )')] 
                kfolds_acc = np.array(kfolds_acc)
                kfolds_feat_num = np.array(kfolds_feat_num)
                #print kfolds_res
                print kfolds_acc
                tplt_l.append([kfolds_acc.mean(), kfolds_feat_num.mean(), centrds_num])
                print kfolds_acc.mean(), kfolds_feat_num.mean()
        tplt_ar = np.array(tplt_l)
        plt.xlabel( 'Number of Features' )
        plt.ylabel( 'Accuracy' )
        plt.plot(tplt_ar[:,1], tplt_ar[:,0], color[centrds_num] + symbol[centrds_num] + line_type[centrds_num])#,\
                 #label=str(centrds_num)+' centroids - sigma='+str(sigma)) #, tvect_d[ nu ], res_d[ nu ], color[i] + line_type[i], label=nu)
        #plt.legend(loc=4 )
        plt.grid(True)
    plt.Figure()
    plt.show()
                 


def plot_resaults(titles, vformat_d, figure_num):
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ] #['b', 'g', 'm', 'k', 'r', 'g', 'b', 'c', 'y' ] #[ 1, 0.9, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8 ] #
    symbol = [ "*", "^", "x", "+", "o" , "*", "^", "x", "+" ]
    line_type = [ "-", "-", "-", "-", "-" , "--", "--", "--", "--" ]
    ylbl = ['F1', 'Prec', 'Recall', "Accuracy"]
    plt.figure( figure_num )
    plt_pos = 0 
    plt.title( titles[0] )
    for vformat in ['**** Binary ****']: #, '**** Normalised by Max Term ****']: #vformat_d.keys(): #['**** Inverse Binary ****'], '**** Binary ****', '**** Normalised by Max Term ****']: 
        tvect_d, f1_d, prec_d, recl_d, acc_d = vformat_d[ vformat ]
        for k, res_d in enumerate([f1_d, prec_d, recl_d]): #f1_d, prec_d, recl_d, acc_d 
            plt_pos += 1      
            plt.subplot(3,1, plt_pos )
            if k == 0:
                plt.title( titles[0].replace('----', '') + vformat.replace('****', '') )
            #plt.xlabel( 'Vects Number for Trainning' )
            plt.xlabel( 'Number of Features kept' )
            #plt.xlabel( 'Frequency Threshold' )
            plt.ylabel( ylbl[k] )
            for i, nu in enumerate(['0.05', '0.07', '0.1', '0.15', '0.2', '0.3', '0.5', '0.7', '0.8']): #['2', '10']): #['0.05', '0.07', '0.1', '0.15', '0.2', '0.3', '0.5', '0.7', '0.8']): #['2', '10']): # ['1', '2', '5', '10', '50']): '1', '2', '5', '10', '50']): #[0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8]): #[0.2, 0.3, 0.5, 0.7, 0.8]): 
                #print len(tvect_d[nu]), len(res_d[nu]), res_d[nu]
                plt.plot(tvect_d[ nu ], res_d[ nu ], color[i] + symbol[i] + line_type[i], label=nu) #, tvect_d[ nu ], res_d[ nu ], color[i] + line_type[i], label=nu)
                plt.legend(loc=4 )
                plt.grid(True)
            plt.Figure()
    return plt

if __name__ == '__main__':
    
    featr_size_lst = [100,300,500,700,1000,3000,5000,10000,15000,20000,30000]
    Centroids_ll = [ 
                     #[0.5],\
                     #[0.2, 0.3],\
                     [0.2, 0.5, 0.8],\
                     #[0.2, 0.4, 0.6, 0.8],\
                     [0.1, 0.3, 0.5, 0.7, 0.9],\
                     #[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                   ] 
    Sigma_l = [0.2]
    
    h5f = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words_TM-Derivative(>0).h5', 'r')
    #h5f = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santini_TT-Words_TM-Derivative(+-).h5', 'r')
    
    #h5f = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/CSVM_RES_LowBow_Words_FAST.h5', 'r')
    
    #sv_h5f = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/CSVM_RES_Words_FAST_merged10folds.h5', 'r')
    #sv_table = sv_h5f.createTable(sv_h5f.root, "multiclass_cross_validation",  MultiResultsTable_desc)
   
    #print h5f.root.multiclass_cross_validation
    plot_data(h5f.root.multiclass_cross_validation, "",  featr_size_lst, Centroids_ll, Sigma_l)
    
    h5f.close()
    
""" 
    for z, file in enumerate(flist):
        print file
        titles, vformat_d = load_resaults_2(base_filepath, file)
        #print vformat_d['**** Binary ****']
        print "Binary", max_tuple(vformat_d['**** Binary ****'][0], vformat_d['**** Binary ****'][1])
        print "TF Normalised by Max Term", max_tuple(vformat_d['**** Normalised by Max Term ****'][0], vformat_d['**** Normalised by Max Term ****'][1])
        plt = plot_resaults(titles, vformat_d, z)
        plot_l.append( plt )
    for plt in plot_l:
        plt.show()
"""
    
    
    
    
    
    
    
    
       