"""

"""
import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')

import numpy as np
import tables as tb
#import html2tf.tables.cngrams as cng_tb
#import html2vect.base.convert.tfttools as tbtls
#from html2vect.base.convert.tfdtools import TFDictTools
#from html2vect.base.convert.convert import TFVects2Matrix2D

import sklearn.decomposition as decomp
import sklearn.svm.libsvm_sparse as libsvm
import sklearn.svm as svm
import sklearn.svm.sparse as sp_svm
import sklearn.linear_model as linear_model
import sklearn.covariance as skcov
from sklearn.metrics import precision_score, recall_score
from sklearn import cross_validation

import scipy.sparse as ssp
from scipy import stats

from html2vect.sparse.lowbow import Html2LBN, Html2LBW, Html2LBN4SEG, Html2LBN_L1_BW
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



class CSVM_CrossVal_Lowbow(object):
    
    def __init__(self, lowbow_type, h5f_res, corpus_path, genres):        
        self.lowbow = Html2LBW(attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm) #lowbow_type 
        self.corpus_path = corpus_path
        self.genres_lst = genres
        self.res_table = h5f_res.createTable(h5f_res.root, "multiclass_cross_validation",  MultiResultsTable_desc)
    
        
    def corpus_files_and_tags(self):
        xhtml_file_l = list()
        cls_gnr_tgs = list()
        for i, g in enumerate(genres):
            gnrs_file_lst = self.lowbow.file_list_frmpaths(self.corpus_path, [ str( g + "/html/" ) ] )
            
            xhtml_file_l.extend( gnrs_file_lst )
                 
            cls_gnr_tgs.extend( [i+1]*len(gnrs_file_lst) )
                
        return (xhtml_file_l, cls_gnr_tgs)
    
    
    def evaluate(self, xhtml_file_l, cls_gnr_tgs, kfolds, C_lst, featr_size_lst, Kernel, Centroids_ll, Sigma_l):
        #Define SVM type
        
        #csvm = svm.LinearSVC(C=c)
        
        #Convert lists to Arrays
        xhtml_file_l = np.array( xhtml_file_l )
        cls_gnr_tgs = np.array( cls_gnr_tgs )
        
        #Starting CrossValidation
        KF = cross_validation.StratifiedKFold(cls_gnr_tgs, kfolds, indices=True)
        for k, (trn_idxs, crv_idxs) in enumerate(KF):
            
            print "Creating DICTIONARY "
            tf_d = dict() 
            #Merge All Term-Frequency Dictionaries created by the Raw Texts            
            for html_str in self.lowbow.load_files( list( xhtml_file_l[trn_idxs] ), encoding='utf8', error_handling='replace' ):
                tf_d = self.lowbow.merge_tfds(tf_d, self.lowbow.tf_dict( self.lowbow._attrib_(html_str) ) )
            
            #SELECT FEATURE SIZE 
            for featrs_size in featr_size_lst:
                resized_tf_d = self.lowbow.keep_atleast(tf_d, featrs_size) #<---
                #print len(resized_tf_d)
                print resized_tf_d.items()[0:50]
            
                #Create The Terms-Index Dictionary that is shorted by Frequency descending order
                tid = self.lowbow.tf2tidx( resized_tf_d )
                print tid.items()[0:50]
                
                #SELECT CENTROIDS AND SIGMA
                for centroids_l in Centroids_ll:
                    for Sigma in Sigma_l:
                        print "Creating LOWBOW"
                        #Create LowBow Vectors Sparse Matrix
                        corpus_mtrx = self.lowbow.from_files( list( xhtml_file_l ),\
                                                              centroids_l, Sigma, tid_dictionary=tid,\
                                                              encoding='utf8', error_handling='replace' )
                        
                        #SELECT C
                        for c in C_lst:
                            csvm = sp_svm.SVC(C=1, kernel='linear', scale_C=False)
                            #csvm.set_params(C=c, kernel=Kernel, scale_C=False)
                            
                            print "FIT MODEL"
                            #FIT MODEL
                            print trn_idxs
                            train_Y = cls_gnr_tgs[ trn_idxs ]
                            print train_Y 
                            mtrx = corpus_mtrx[0]
                            train_X = mtrx[trn_idxs,:]   
                            #Some checks    
                            print ssp.issparse(train_X), train_X.shape[0], train_X.shape[1] #, len(train_Y), train_Y 
                            print ssp.isspmatrix_csr(train_X)
                            
                            csvm.fit( ssp.csr_matrix(train_X.todense(), shape=train_X.shape), train_Y) #<----- WATCH THIS
                            
                            print "EVALUATE"
                            #EVALUATE
                            print crv_idxs
                            crossval_Y = cls_gnr_tgs[ crv_idxs ]
                            print crossval_Y
                            mtrx = corpus_mtrx[0]
                            crossval_X = mtrx[crv_idxs,:]
                             
                            #ssp.csr_matrix(crossval_X, shape=crossval_X.shape, dtype=np.float64)
                            res_acc_score = csvm.score( ssp.csr_matrix(crossval_X.todense(), shape=crossval_X.shape), crossval_Y) #<----- WATCH THIS
                            
                            print "Accuracy:", res_acc_score, " for:", k, c, len(resized_tf_d), centroids_l, Sigma 
                            
                            print "SAVE RESAULTS"
                             
                            self.res_table.row['kfold'] = k
                            self.res_table.row['C'] = c
                            self.res_table.row['feat_num'] = len(resized_tf_d)
                            self.res_table.row['Acc'] = res_acc_score
                            self.res_table.row.append()
                            self.res_table.flush()
                 
                 

if __name__=='__main__':
    
    lowbow_N = Html2LBN(3, attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm)
    lowbow_N4SG = Html2LBN4SEG(3, attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm)
    lowbow_N4L2 = Html2LBN_L1_BW(3, attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm)
    lowbow_W = Html2LBW(attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm)
    
    #corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
    corpus_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/"
    #genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop" ]
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/CSVM_LOWBOW_RES.h5', 'w')
    crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/CSVM_LOWBOW_RES.h5', 'w')
    
    csvm_crossval_lowbow = CSVM_CrossVal_Lowbow(lowbow_W ,crp_crssvl_res, corpus_filepath, genres)
    
    kfolds = 10
    C_lst = [1]
    featr_size_lst = [15000]
    Kernel = 'linear'
    Centroids_ll = [[0.2, 0.5, 0.8]] 
    Sigma_l = [0.2, 0.5, 0.7]
            
    xhtml_file_l, cls_gnr_tgs = csvm_crossval_lowbow.corpus_files_and_tags()
    
    csvm_crossval_lowbow.evaluate(xhtml_file_l, cls_gnr_tgs, kfolds, C_lst, featr_size_lst, Kernel, Centroids_ll, Sigma_l)
    
    crp_crssvl_res.close()

    
    
    
    
    
    
    
    
    
    
    

