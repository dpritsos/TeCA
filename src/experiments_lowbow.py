"""

"""
import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')

import numpy as np
import tables as tb
#import html2tf.tables.cngrams as cng_tb
import html2vect.base.convert.tfttools as tbtls
from html2vect.base.convert.tfdtools import TFDictTools
from html2vect.base.convert.convert import TFVects2Matrix2D

import sklearn.decomposition as decomp
import sklearn.svm as svm
import sklearn.svm.sparse as sp_svm
import sklearn.linear_model as linear_model
import sklearn.covariance as skcov
from sklearn.metrics import precision_score, recall_score

import scipy.sparse as ssp

from html2vect.sparse.lowbow import Html2LBN 

#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.grid_search import GridSearchCV
  
class ResultsTable_desc(tb.IsDescription):
    kfold = tb.UInt32Col(pos=1)
    nu = tb.Float32Col(pos=2)
    feat_num = tb.UInt32Col(pos=3)
    F1 = tb.Float32Col(pos=4)
    P = tb.Float32Col(pos=5)
    R = tb.Float32Col(pos=6)
    TP = tb.Float32Col(pos=7)
    FP = tb.Float32Col(pos=8)
    TN = tb.Float32Col(pos=9)
    FN = tb.Float32Col(pos=10)
    
class MultiResultsTable_desc(tb.IsDescription):
    kfold = tb.UInt32Col(pos=1)
    C = tb.Float32Col(pos=2)
    feat_num = tb.UInt32Col(pos=3)
    Acc = tb.Float32Col(pos=7)



base_filepath = ["/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/" ]
genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]

for g in genres:
    filepath = str( "/" + g + "/html_pages/")
   

class CSVM_CrossVal(object):
    
    def __init__(self, h5f_res, corpus_path, genres):
        self.h5f_res = h5f_res
        self.corpus_path = corpus_path
        self.genres_lst = genres
        self.kfold_chnk = dict()
        self.page_lst_tb = dict()
        self.kfold_mod = dict()
        self.gnr2clss = dict()
        #self.tfv2matrix2d = TFVects2Matrix2D(DSize=3000)
                    
    def prepare_data(self, kfolds, format):
        
            
            Html2LBN( self.n, lowercase=True, valid_html=True , smoothing_kernel=stats.norm)
        
            training_pg_lst = list()
            training_clss_tag_lst = list()
            crossval_pg_lst = list()
            crossval_clss_tag_lst = list()
            
            for gnr in self.genres_lst:
                page_lst_tb = self.page_lst_tb[gnr].read()
                #Get the Evaluation set for this Genre and later concatenate it to the rest of the Evaluation Genres
                grn_page_lst_tb = page_lst_tb['table_name'][start:end]
                crossval_pg_lst.extend( page_lst_tb['table_name'][start:end] )
                crossval_clss_tag_lst.extend( [self.gnr2clss[gnr]]*len(grn_page_lst_tb) )
                #Get the Training set for this Genre g
                grn_trn_pg_lst = self.complementof_list( page_lst_tb['table_name'], start, end )
                training_pg_lst.extend( grn_trn_pg_lst )
                training_clss_tag_lst.extend( [self.gnr2clss[gnr]]*len(grn_trn_pg_lst) )
                  
                   
            csvm = sp_svm.SVC(kernel='linear', C=1)
            
            train = self.tfv2matrix2d.from_tables(self.h5file, self.corpus_grp, training_pg_lst, data_type=tbtls.default_TF_3grams_dtype)
            
            #
            csvm.fit( np.divide(train, train),\
                      training_clss_tag_lst)
            
            cross = self.tfv2matrix2d.from_tables(self.h5file, self.corpus_grp, crossval_pg_lst, data_type=tbtls.default_TF_3grams_dtype)
            
            #
            print csvm.score( np.divide(cross, cross),\
                             crossval_clss_tag_lst)
            
            self.tfv2matrix2d.Dictionary = None
            
            
    def evaluate(self, kfolds, C_lst, featr_size_lst):
        
        #Create Results table for all this CrossValidation Multi-class SVM
        print "Create Results table for all this CrossValidation Multi-class SVM"
        self.h5f_res.createTable(self.h5f_res.root, "MultiClass_CrossVal",  MultiResultsTable_desc)
        
        for k in range(kfolds):
            #Get the Training-set Dictionary - Sorted by frequency for THIS-FOLD
            print "Get Dictionary - Sorted by Frequency for this kfold:", k
            kfold_Dictionary_TF_arr = self.h5f_data.getNode(self.h5f_data.root, 'kfold_'+str(k)+'_Dictionary_TF_arr')
            
            #Get Training-set for kFold
            print "Get Training Data Array for kfold:", k
            training_earr_X = self.h5f_data.getNode(self.h5f_data.root, 'kfold_'+str(k)+'_Training_X')
            training_earr_Y = self.h5f_data.getNode(self.h5f_data.root, 'kfold_'+str(k)+'_Training_Y')
            
            #Get the Evaluation-set for kfold
            print "Get Evaluation Data Array for kfold:", k
            crossval_earr_X = self.h5f_data.getNode(self.h5f_data.root, 'kfold_'+str(k)+'_CrossVal_X')
            crossval_earr_Y = self.h5f_data.getNode(self.h5f_data.root, 'kfold_'+str(k)+'_CrossVal_Y')
            
            #Get Results table for all this CrossValidation Multi-class SVM
            print "Get Results table for all this CrossValidation Multi-class SVM"
            res_table = self.h5f_res.getNode(self.h5f_res.root, "MultiClass_CrossVal")
            
            for featrs_size in featr_size_lst: 
                ##### FIND MORE OPTICAML USE IF POSIBLE
                #Keep the amount of feature required - it will keep_at_least as many as
                feat_len = np.max(np.where(  kfold_Dictionary_TF_arr.read()['freq'] == kfold_Dictionary_TF_arr.read()['freq'][featrs_size] )[0])
                #the featrs_size keeping all the terms with same frequency the last term satisfies the featrs_size
                print "Features Size:", feat_len      
                for c in C_lst:
                    csvm = svm.SVC(C=c, kernel='linear')
                    #csvm = svm.LinearSVC(C=c)
                    #csvm = sp_svm.LinearSVC(C=c)
                    #csvm = linear_model.SGDClassifier(n_iter=50, alpha=1e-5, n_jobs=1)
                    print "FIT model"
                    ##train_X = training_earr_X[:, 0:feat_len] 
                    train_Y = training_earr_Y[:]
                    train_X = np.where( training_earr_X[::20, 0:feat_len] > 0, training_earr_X[::20, 0:feat_len], 0)
                    train_X[ np.nonzero(train_X) ] = 1
                    
                    #train_X = self.Arr2CsrMtrx( training_earr_X, len(training_earr_X), feat_len )
                    
                    #print train_X
                                                                        
                    csvm.fit(train_X, train_Y)

                    print "Predict for kfold:k",k
                    #crossval_X = crossval_earr_X[:, 0:feat_len] 
                    crossval_Y = crossval_earr_Y[:]
                    crossval_X = np.where( crossval_earr_X[:, 0:feat_len] > 0, crossval_earr_X[:, 0:feat_len], 0)
                    crossval_X[ np.nonzero(crossval_X) ] = 1  
                    res_acc_score = csvm.score(crossval_X, crossval_Y)
                    
                    print "Accuracy:", res_acc_score 
                    res_table.row['kfold'] = k
                    res_table.row['C'] = c
                    res_table.row['feat_num'] = feat_len
                    res_table.row['Acc'] = res_acc_score
                    res_table.row.append()
            res_table.flush()
         
    def exec(self):
        self.prepare_data(10, None)
        self.evaluate(10, [1], [30000])
    
     def complementof_list(self, lst, excld_dwn_lim, excld_up_lim):
        if excld_dwn_lim == 0:
            return lst[excld_up_lim:]
        if excld_up_lim == len(lst):
            return lst[0:excld_dwn_lim]
        inv_lst = np.concatenate((lst[0:excld_dwn_lim], lst[excld_up_lim:]))
        return inv_lst
        

if __name__=='__main__':
    
    kfolds = 10
    nu_lst = [0.2, 0.8]
    featr_size_lst = [1000]
    crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/CSVM_LOWBOW_RES.h5', 'w')
            
    csvm_crossval = CSVM_CrossVal(crp_tftb_h5, crp_crssvl_data, crp_crssvl_res, "/Automated_Crawled_Corpus", "/trigrams/")
    csvm_crossval.exec()
    
    
    crp_tftb_h5.close() 
    crp_crssvl_data.close()
    crp_crssvl_res.close()

    
    
    
    
    
    
    
    
    
    
    

