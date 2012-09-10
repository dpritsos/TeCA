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
from trainevaloneclssvm import SVMTE



class OCSVM_CrossVal(tbtls.TFTablesTools):
    
    def __init__(self, h5file, h5f_data, h5f_res, corpus_grp, trms_type_grp):
        tbtls.TFTablesTools.__init__(self, h5file)
        self.h5file = h5file
        self.h5f_data = h5f_data
        self.h5f_res = h5f_res
        self.corpus_grp = corpus_grp
        self.trms_type_grp = trms_type_grp
        self.genres_lst = list()
        self.kfold_chnk = dict()
        self.page_lst_tb = dict()
        self.kfold_mod = dict()
        self.gnr2clss = dict()
        self.e_arr_filters = tb.Filters(complevel=5, complib='zlib')
        self.ocsvm_gnr_end_bound_dct = dict() 
    
    def complementof_list(self, lst, excld_dwn_lim, excld_up_lim):
        if excld_dwn_lim == 0:
            return lst[excld_up_lim:]
        if excld_up_lim == len(lst):
            return lst[0:excld_dwn_lim]
        inv_lst = np.concatenate((lst[0:excld_dwn_lim], lst[excld_up_lim:]))
        return inv_lst
                    
    def prepare_data(self, kfolds, format):
        corpus = self.h5file.getNode('/', self.corpus_grp)
        self.genres_lst = corpus._v_attrs.genres_lst
        for i, gnr in enumerate(self.genres_lst):
            self.page_lst_tb[gnr] = self.h5file.getNode( self.corpus_grp + self.trms_type_grp + gnr, '/PageListTable' )
            self.kfold_chnk[gnr] = np.divide(len(self.page_lst_tb[gnr]),kfolds)
            self.kfold_mod[gnr] = np.divide(len(self.page_lst_tb[gnr]),kfolds)
            self.gnr2clss[gnr] = i + 1
        
        for k in range(kfolds):
            start = k*self.kfold_chnk[gnr]
            end = (k+1)*self.kfold_chnk[gnr]
            if (k+1) == kfolds:
                end = end + self.kfold_mod[gnr]
            
            for g in self.genres_lst:
                training_pg_lst = list()
                training_clss_tag_lst = list()
                crossval_pg_lst = list()
                crossval_clss_tag_lst = list()
                
                page_lst_tb = self.page_lst_tb[g].read()
                
                #Get the Training set for this Genre g
                grn_trn_pg_lst = self.complementof_list( page_lst_tb['table_name'], start, end )
                training_pg_lst.extend( grn_trn_pg_lst )
                training_clss_tag_lst.extend( [self.gnr2clss[g]]*len(grn_trn_pg_lst) )
                
                #Get the Evaluation set for this Genre and later concatenate it to the rest of the Evaluation Genres
                grn_page_lst_tb = page_lst_tb['table_name'][start:end]
                crossval_pg_lst.extend( page_lst_tb['table_name'][start:end] )
                crossval_clss_tag_lst.extend( [self.gnr2clss[g]]*len(grn_page_lst_tb) )
                self.ocsvm_gnr_end_bound_dct[g] = end
                for gnr in self.genres_lst:
                    if gnr != g:
                        page_lst_tb = self.page_lst_tb[gnr].read()
                        grn_page_lst_tb = page_lst_tb['table_name'][:]
                        crossval_pg_lst.extend( page_lst_tb['table_name'][:] )
                        crossval_clss_tag_lst.extend( [self.gnr2clss[gnr]]*len(grn_page_lst_tb) )
                    
                print len(crossval_pg_lst)
                print len(crossval_clss_tag_lst), crossval_clss_tag_lst 
                print len(training_pg_lst), 
                print len(training_clss_tag_lst), training_clss_tag_lst 
                
                #Create the Training-set Dictionary - Sorted by frequency
                print "Creating Dictionary - Sorted by Frequency - for ", g 
                term_idx_d, kfold_Dictionary_TF_arr = self.TFTables2TFDict_n_TFArr(self.corpus_grp + self.trms_type_grp,\
                                                                                   training_pg_lst,\
                                                                                   data_type=np.dtype([('terms', 'S128'), ('freq', 'float32')]))
                
                #print term_idx_d.items()[0:10]
                #print kfold_Dictionary_TF_arr[0:10] 
                
                DicFreq = self.h5f_data.createTable(self.h5f_data.root, g+'_kfold_'+str(k)+'_Dictionary_TF_arr', kfold_Dictionary_TF_arr)
                
                print DicFreq[0:10]
                
                ###############From this line and down need to be veryfied that the reasaults are corect!#############
                
                #Create the Training-Set Array
                print "Preparing Training Set for", g
                training_earr_X = self.h5f_data.createEArray(self.h5f_data.root, g+'_kfold_'+str(k)+'_Training_X',\
                                                      tb.Float32Atom(), (0,len(term_idx_d)),\
                                                      filters=self.e_arr_filters)
                training_earr_X = self.TFTabels2EArray(training_earr_X, self.corpus_grp + self.trms_type_grp,
                                                     training_pg_lst, term_idx_d,\
                                                     data_type=np.float32)
                training_earr_Y = self.h5f_data.createArray(self.h5f_data.root, g+'_kfold_'+str(k)+'_Training_Y', np.array(training_clss_tag_lst))
                                                                         
                
                print training_earr_X[0:5, 10:20]
                print np.shape(training_earr_X)
                print training_earr_Y
                
                #Create the CrossVal-Set Array
                print "Preparing Cross-Validation Set for", g
                crossval_earr_X = self.h5f_data.createEArray(self.h5f_data.root, g+'_kfold_'+str(k)+'_CrossVal_X',\
                                                      tb.Float32Atom(), (0,len(term_idx_d)),\
                                                      filters=self.e_arr_filters)
                crossval_earr_X = self.TFTabels2EArray(crossval_earr_X, self.corpus_grp + self.trms_type_grp,
                                                     crossval_pg_lst, term_idx_d,\
                                                     data_type=np.float32)       
                crosval_earr_Y = self.h5f_data.createArray(self.h5f_data.root, g+'_kfold_'+str(k)+'_CrossVal_Y', np.array(crossval_clss_tag_lst))
                
                print crossval_earr_X[0:5, 10:20]
                print np.shape(crossval_earr_X)
                print crosval_earr_Y
                
            
    def evaluate(self, kfolds, nu_lst, featr_size_lst, end_dct):
    
        self.genres_lst = end_dct
        
        self.ocsvm_gnr_end_bound_dct = end_dct
        
        for g in self.genres_lst:
            #Create Results table for all this CrossValidation Multi-class SVM
            print "Create Results table for all this CrossValidation Multi-class SVM for genre:", g
            self.h5f_res.createTable(self.h5f_res.root, g+"_MultiClass_CrossVal",  ResultsTable_desc)
            
            for k in range(kfolds):
                #Get the Training-set Dictionary - Sorted by frequency for THIS-FOLD
                print "Get Dictionary - Sorted by Frequency for this kfold:", k
                kfold_Dictionary_TF_arr = self.h5f_data.getNode(self.h5f_data.root, g+'_kfold_'+str(k)+'_Dictionary_TF_arr')
                
                #Get Training-set for kFold
                print "Get Training Data Array for kfold & genre:", k, g
                training_earr_X = self.h5f_data.getNode(self.h5f_data.root, g+'_kfold_'+str(k)+'_Training_X')
                training_earr_Y = self.h5f_data.getNode(self.h5f_data.root, g+'_kfold_'+str(k)+'_Training_Y')
                
                #Get the Evaluation-set for kfold
                print "Get Evaluation Data Array for kfold & genre:", k, g
                crossval_earr_X = self.h5f_data.getNode(self.h5f_data.root, g+'_kfold_'+str(k)+'_CrossVal_X')
                crossval_earr_Y = self.h5f_data.getNode(self.h5f_data.root, g+'_kfold_'+str(k)+'_CrossVal_Y')
                
                #Get Results table for all this CrossValidation Multi-class SVM
                print "Get Results table for all this CrossValidation Multi-class SVM and gerne:", g
                res_table = self.h5f_res.getNode(self.h5f_res.root, g+'_MultiClass_CrossVal')
                
                for featrs_size in featr_size_lst: 
                    ##### FIND MORE OPTICAML USE IF POSIBLE
                    #Keep the amount of feature required - it will keep_at_least as many as
                    feat_len = np.max(np.where(  kfold_Dictionary_TF_arr.read()['freq'] == kfold_Dictionary_TF_arr.read()['freq'][featrs_size] )[0])
                    #the featrs_size keeping all the terms with same frequency the last term satisfies the featrs_size
                    print "Features Size:", feat_len      
                    for n in nu_lst:
                        ocsvm = svm.OneClassSVM(nu=n, kernel='rbf')
                        #tfidf = TfidfTransformer()
                        #ppca = decomp.ProbabilisticPCA(n_components=5)
                        #ecov = skcov.EmpiricalCovariance() 
                        #sgd = linear_model.SGDClassifier(n_iter=50, alpha=1e-5, n_jobs=1)
                        print "FIT model"
                        #train_X = training_earr_X[0:180, 0:feat_len] 
                        ## No Required for this implementation train_Y = training_earr_Y[:]
                        train_X = np.where( training_earr_X[:, 0:feat_len] > 0, training_earr_X[:, 0:feat_len], 0)
                        #train_X = tfidf.fit_transform(train_X)
                        #print train_X.todense()
                        #train_X[ np.nonzero(train_X) ] = 1  
                        train_X = np.divide(train_X, np.amax( train_X, axis=1 ).reshape(len(train_X), 1) )
                        #print len(train_X), len(train_Y)
                        #print train_X, train_Y
                        #0/0
                        #sgd.fit(train_X, train_Y)
                        #ecov.fit(train_X)
                        #ppca.fit(train_X)  
                        ocsvm.fit(train_X) 
                        print "Predict for kfold:k", k
                        #crossval_X = crossval_earr_X[:, 0:feat_len] 
                        #print len(crossval_X) 
                        
                        ## No Required for this implementation crossval_Y = crossval_earr_Y[:]
                        crossval_Y = crossval_earr_Y[:]
                        crossval_Y[0:self.ocsvm_gnr_end_bound_dct[g]] = 1
                        crossval_Y[self.ocsvm_gnr_end_bound_dct[g]::] = -1
                        crossval_X = np.where( crossval_earr_X[:, 0:feat_len] > 0, crossval_earr_X[:, 0:feat_len], 0)
                        #crossval_X = tfidf.fit_transform(crossval_X)
                        #crossval_X[ np.nonzero(crossval_X) ] = 1
                        crossval_X = np.divide( crossval_X,  np.amax( crossval_X, axis=1 ).reshape(len(crossval_X), 1) )
                        #print crossval_X 
                        res0 = ocsvm.predict(crossval_X[0:self.ocsvm_gnr_end_bound_dct[g]])
                        #res = ppca.score(crossval_X[0:self.ocsvm_gnr_end_bound_dct[g]])
                        #res = ecov.mahalanobis(crossval_X[0:self.ocsvm_gnr_end_bound_dct[g]])
                        #res = sgd.predict(crossval_X[0:self.ocsvm_gnr_end_bound_dct[g]])
            
                        tp = len( np.where( res0 == 1 )[0] )
                        fn = len( np.where( res0 == -1 )[0] )
                        res1 = ocsvm.predict(crossval_X[self.ocsvm_gnr_end_bound_dct[g]::])
                        #res = ppca.score(crossval_X[self.ocsvm_gnr_end_bound_dct[g]::])
                        #res = ecov.mahalanobis(crossval_X[self.ocsvm_gnr_end_bound_dct[g]::])
                        #res = sgd.predict(crossval_X[self.ocsvm_gnr_end_bound_dct[g]::])
                        tn = len( np.where( res1 == -1 )[0] ) 
                        fp = len( np.where( res1 == 1 )[0] )
                                                
                        if (tp + fp) != 0.0:
                            P = np.float(tp) / np.float( tp + fp )
                        else:
                            P = 0.0
                        if (tp + fn) != 0.0:
                            R = np.float(tp) / np.float( tp + fn )
                        else:
                            R = 0.0
                        if (P + R) != 0.0:
                            F1 = np.float( 2 * P * R ) / np.float( P + R )
                        else:
                            F1 = 0.0
                        if (tp + tn + fp + fn) !=0:
                            Acc = np.float(tp + tn) / np.float(tp + tn + fp + fn)
                        else:
                            Acc = 0.0
                        print tp, fp
                        print fn, tn 
                        print P, precision_score( crossval_Y, np.hstack((res0, res1)) ), R, recall_score( crossval_Y, np.hstack((res0, res1)) ), F1 
                        res_table.row['kfold'] = k
                        res_table.row['nu'] = n
                        res_table.row['feat_num'] = feat_len
                        res_table.row['F1'] = F1
                        res_table.row['P'] = P
                        res_table.row['R'] = R
                        res_table.row['TP'] = tp
                        res_table.row['FP'] = fp
                        res_table.row['FN'] = fn
                        res_table.row['TN'] = tn
                        res_table.row.append()  
                        
                res_table.flush()
            
    def exec_test(self):
        self.prepare_data(10, None)
        end_dct = { "eshop":20, "blog":20, "faq":20, "frontpage":20, "listing":20, "php":20, "spage":20}
        self.evaluate(10, [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.80, 0.9, 0.95], [30, 300, 3000], end_dct)
        #self.evaluate(10, [1], [30, 300, 3000], end_dct)


if __name__=='__main__':
    
    kfolds = 10
    nu_lst = [0.2, 0.8]
    featr_size_lst = [1000]
    crp_tftb_h5 = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Santini_corpus.h5', 'r')
    #crp_tftb_h5 = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/ACC.h5', 'r')
    crp_crssvl_data = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/CSVM_CrossVal_Data_TEST_SPARSE.h5', 'w')
    #crp_crssvl_data = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/CSVM_CrossVal_Data_TEST_SPARSE.h5', 'w')
    crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/CSVM_CrossVal_Results_TEST_SPASE.h5', 'w')
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/CSVM_CrossVal_Results_TEST_SPARSE.h5', 'w')
    
    #crp_tftb_h5 = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Santini_corpus_w.h5', 'r')
    #crp_tftb_h5 = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/ACC.h5', 'r')
    #crp_crssvl_data = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_CrossVal_Data_w.h5', 'r')
    #crp_crssvl_data = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/CSVM_CrossVal_Data.h5', 'r')
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_CrossVal_Results_w.h5', 'w')
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/CSVM_CrossVal_Results.h5', 'w')
    #genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
    
    #ocsvm_crossval = OCSVM_CrossVal(crp_tftb_h5, crp_crssvl_data, crp_crssvl_res, "/Santini_corpus", "/words/")
    ocsvm_crossval = OCSVM_CrossVal(crp_tftb_h5, crp_crssvl_data, crp_crssvl_res, "/Santini_corpus", "/trigrams/")
    #csvm_crossval = CSVM_CrossVal(crp_tftb_h5, crp_crssvl_data, crp_crssvl_res, "/Santini_corpus", "/trigrams/")
    #csvm_crossval = CSVM_CrossVal(crp_tftb_h5, crp_crssvl_data, crp_crssvl_res, "/Automated_Crawled_Corpus", "/trigrams/")
    #csvm_crossval.exec_test()
    ocsvm_crossval.exec_test()
    
    crp_tftb_h5.close() 
    crp_crssvl_data.close()
    crp_crssvl_res.close()
    
import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
import numpy as np
import tables as tb
#import html2tf.tables.cngrams as cng_tb

import scipy.sparse as ssp
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn import cross_validation

import html2vect.sparse.words as h2v_w
import html2vect.sparse.cngrams as h2v_cng


        
class CrossVal_Koppels_method(object):
    
    def __init__(self, TF_TT, h5_res, corpus_path, genres):
        self.TF_TT = TF_TT
        self.corpus_path = corpus_path
        self.genres_lst = genres
        self.gnrs_num = len(genres)
        self.h5_res = h5_res
        
                
    def corpus_files_and_tags(self):
        xhtml_file_l = list()
        cls_gnr_tgs = list()
        for i, g in enumerate(self.genres_lst):
            gnrs_file_lst = self.TF_TT.file_list_frmpaths(self.corpus_path, [ str( g + "/html/" ) ] )
            
            xhtml_file_l.extend( gnrs_file_lst )
                 
            cls_gnr_tgs.extend( [i+1]*len(gnrs_file_lst) )
                
        return (xhtml_file_l, cls_gnr_tgs)
    
                      
    def contruct_classes(self, trn_idxs, corpus_mtrx, cls_gnr_tgs):
        inds_per_gnr = dict()
        inds = list()
        last_gnr_tag = 1
        
        for trn_idx in trn_idxs:
            
            if cls_gnr_tgs[trn_idx] != last_gnr_tag:
                inds_per_gnr[ self.genres_lst[last_gnr_tag - 1] ] = inds
                last_gnr_tag = cls_gnr_tgs[trn_idx]
                inds = []
            
            inds.append( trn_idx )    
        
        inds_per_gnr[ self.genres_lst[last_gnr_tag - 1] ] = inds 
    
        gnr_classes = dict()
        for g in self.genres_lst:
            #Merge All Term-Frequency Dictionaries created by the Raw Texts
            gnr_classes[g] = corpus_mtrx[inds_per_gnr[g], :].sum(axis=0)
        
        return (gnr_classes, inds_per_gnr)   
    
    
    def predict(self, gnr_classes, crossval_X, crossval_Y, vocab_index_dct, featrs_size, similarity_func, sim_min_value, iters, sigma_threshold):
            
        max_sim_scores_per_iter = np.zeros((iters, crossval_X.shape[0]))
        predicted_classes_per_iter = np.zeros((iters, crossval_X.shape[0]))
                    
        #Measure similarity for iters iterations i.e. for iters different feature subspaces Randomly selected 
        for I in range(iters):
            
            #Randomly select some of the available features
            suffled_vocabilary_idxs = np.random.permutation( np.array(vocab_index_dct.values()) ) 
            features_subspace = suffled_vocabilary_idxs[0:featrs_size]
            
            #Initialised Predicted Classes and Maximum Similarity Scores Array for this i iteration 
            predicted_classes = np.zeros( crossval_X.shape[0] )
            max_sim_scores = np.zeros( crossval_X.shape[0] )
            
            #Measure similarity for each Cross-Validation-Set vector to each available Genre Class(i.e. Class-Vector). For This feature_subspace
            for i_vect, vect in enumerate(crossval_X[:, features_subspace]):
    
                #NOTE: max_sim is initialised for cosine-similarity which ranges from -1 to 1, with 1 indicates maximum similarity
                #However, it can be any value given by sim_min_value argument  
                max_sim = sim_min_value
                for cls_tag, g in enumerate(self.genres_lst):
                    
                    #Measure Similarity
                    sim_score = similarity_func(vect, gnr_classes[ g ][:, features_subspace])
                    
                    #Just for debugging for 
                    if sim_score > 1.0:
                        print "FUCK"
                    
                    if sim_score > max_sim:
                        predicted_classes[i_vect] = cls_tag + 1 #plus 1 is the real class tag 0 means uncategorised
                        max_sim_scores[i_vect] = sim_score
                        max_sim = sim_score
        
            #Store Predicted Classes and Scores for this i iteration
            max_sim_scores_per_iter[I,:] = max_sim_scores[:]
            predicted_classes_per_iter[I,:] = predicted_classes[:]
                                              
        predicted_Y = np.zeros((crossval_Y.shape[0]), dtype=np.float)
        predicted_scores = np.zeros((crossval_Y.shape[0]), dtype=np.float)
        
        for i_prd_cls, prd_cls in enumerate(predicted_classes_per_iter.transpose()):
            genres_occs = np.histogram( prd_cls.astype(np.int), bins=np.arange(self.gnrs_num+2))[0] #One Bin per Genre plus one i.e the first to be always zero
            #print genres_occs
            genres_probs = genres_occs.astype(np.float) / np.float(iters)
            #print genres_probs
            #if np.max(genres_probs) >= sigma_threshold:
            predicted_Y[i_prd_cls] = np.argmax( genres_probs )
            predicted_scores[i_prd_cls] = np.max( genres_probs ) 
        
        return predicted_Y, predicted_scores, max_sim_scores_per_iter, predicted_classes_per_iter      
        
    
    def evaluate(self, xhtml_file_l, cls_gnr_tgs, kfolds, vocabilary_size, iter_l, featr_size_lst, sigma_threshold, similarity_func, norm_func):
        
        #Convert lists to Arrays
        xhtml_file_l = np.array( xhtml_file_l )
        cls_gnr_tgs = np.array( cls_gnr_tgs )
        
        #Starting CrossValidation
        KF = cross_validation.StratifiedKFold(cls_gnr_tgs, kfolds, indices=True)
        for k, (trn_idxs, crv_idxs) in enumerate(KF):
            
            #Creating a Group for this k-fold in h5 file
            kfld_group = self.h5_res.createGroup('/', 'KFold'+str(k), "K-Fold group of Results Arrays" )
            
            print "Creating DICTIONARY "
            tf_d = dict() 
            #Merge All Term-Frequency Dictionaries created by the Raw Texts            
            for html_str in self.TF_TT.load_files( list( xhtml_file_l[trn_idxs] ), encoding='utf8', error_handling='replace' ):
                tf_d = self.TF_TT.merge_tfds(tf_d, self.TF_TT.tf_dict( self.TF_TT._attrib_(html_str) ) )
                 
            #SELECT VOCABILARY SIZE 
            for vocab_size in vocabilary_size:
                resized_tf_d = self.TF_TT.keep_atleast(tf_d, vocab_size) #<---
                print len(resized_tf_d)
                print resized_tf_d.items()[0:50]
                
                #Create The Terms-Index Dictionary that is shorted by Frequency descending order
                tid = self.TF_TT.tf2tidx( resized_tf_d )
                print tid.items()[0:50]
                
                print "Creating Sparse TF Matrix for CrossValidation"
                #Create Sparse TF Vectors Sparse Matrix
                corpus_mtrx = self.TF_TT.from_files( list( xhtml_file_l ), tid_dictionary=tid, norm_func=norm_func,\
                                                         encoding='utf8', error_handling='replace' )
                
                print "Construct classes"
                #Construct Genres Class Vectors form Training Set
                gnr_classes, inds_per_gnr = self.contruct_classes(trn_idxs, corpus_mtrx[0], cls_gnr_tgs)
                
                #SELECT Cross Validation Set
                crossval_Y = cls_gnr_tgs[ crv_idxs ]
                mtrx = corpus_mtrx[0]
                crossval_X = mtrx[crv_idxs, :] 
                                
                #SELECT FREATUR SIZE
                for featrs_size in featr_size_lst:
                    
                    #Creating a Group for this features size in h5 file under this k-fold
                    feat_num_group = self.h5_res.createGroup(kfld_group, 'Feat'+str(featrs_size), "Features Number group of Results Arrays for this K-fold" )
                    
                    #SELECT DIFERENT ITERATIONS NUMBER
                    for iters in iter_l:
                        print "EVALUATE"
                        
                        #Creating a Group for this number of iterations in h5 file under this features number under this k-fold
                        iters_group = self.h5_res.createGroup(feat_num_group, 'Iters'+str(iters), "Number of Iterations (for statistical prediction) group of Results Arrays for this K-fold" )
                       
                        predicted_Y,\
                        predicted_scores,\
                        max_sim_scores_per_iter,\
                        predicted_classes_per_iter = self.predict(gnr_classes,\
                                                                  crossval_X, crossval_Y,\
                                                                  tid, featrs_size,\
                                                                  similarity_func, -1.0,\
                                                                  iters, sigma_threshold) 
                        
                        
                        
                        print np.histogram(crossval_Y, bins=np.arange(self.gnrs_num+2))
                        print np.histogram(predicted_Y.astype(np.int), bins=np.arange(self.gnrs_num+2))
                        
                        cv_tg_idxs = np.array( np.histogram(crossval_Y, bins=np.arange(self.gnrs_num+2))[0], dtype=np.float)
                        tp_n_fp = np.array( np.histogram(predicted_Y.astype(np.int), bins=np.arange(self.gnrs_num+2))[0], dtype=np.float)
                        
                        P_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                        R_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                        F1_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                        
                        end = 0
                        for gnr_cnt in range(len(self.genres_lst)):
                            start = end
                            end = end + cv_tg_idxs[gnr_cnt+1]
                            counts_per_grn_cv = np.histogram( predicted_Y[start:end], bins=np.arange(self.gnrs_num+2) )[0]
                            #print counts_per_grn_cv
                            #print tp_n_fp[gnr_cnt+1]
                            P = counts_per_grn_cv.astype(np.float) / tp_n_fp[gnr_cnt+1]
                            P_per_gnr[gnr_cnt+1] = P[gnr_cnt+1]
                            R = counts_per_grn_cv.astype(np.float) / cv_tg_idxs[gnr_cnt+1]
                            R_per_gnr[gnr_cnt+1] = R[gnr_cnt+1]  
                            F1_per_gnr[gnr_cnt+1] = 2 * P[gnr_cnt+1] * R[gnr_cnt+1] / (P[gnr_cnt+1] + R[gnr_cnt+1]) 
                            
                        P_per_gnr[0] = precision_score(crossval_Y, predicted_Y)   
                        R_per_gnr[0] = recall_score(crossval_Y, predicted_Y) 
                        F1_per_gnr[0] = f1_score(crossval_Y, predicted_Y)  
                        
                        #Maybe Later
                        #fpr, tpr, thresholds = roc_curve(crossval_Y, predicted_Y)   
                                            
                        print self.h5_res.createArray(iters_group, 'predicted_Y', predicted_Y, "predicted Classes per Document (CrossValidation Set)")[:]
                        print self.h5_res.createArray(iters_group, 'predicted_classes_per_iter', predicted_classes_per_iter, "Predicted Classes per Document per Iteration (CrossValidation Set)")[:]
                        print self.h5_res.createArray(iters_group, 'predicted_scores', predicted_scores, "predicted Scores per Document (CrossValidation Set)")[:]
                        print self.h5_res.createArray(iters_group, 'max_sim_scores_per_iter', max_sim_scores_per_iter, "Max Similarity Score per Document per Iteration (CrossValidation Set)")[:]                        
                        print self.h5_res.createArray(iters_group, "P_per_gnr", P_per_gnr, "Precision per Genre (P[0]==Global P)")[:]
                        print self.h5_res.createArray(iters_group, "R_per_gnr", R_per_gnr, "Recall per Genre (R[0]==Global R)")[:]
                        print self.h5_res.createArray(iters_group, "F1_per_gnr", F1_per_gnr, "F1_statistic per Genre (F1[0]==Global F1)")[:]
                        print                
                                        
               

def cosine_similarity(vector, centroid):
 
    return vector * np.transpose(centroid) / ( np.linalg.norm(vector.todense()) * np.linalg.norm(centroid) )



if __name__ == '__main__':
    
    sparse_W = h2v_w.Html2TF(attrib='text', lowercase=True, valid_html=False)
    sparse_CNG = h2v_cng.Html2TF(4, attrib='text', lowercase=True, valid_html=False)
    
    corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
    #corpus_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/"
    genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    #genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop" ]
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santini_TT-Words_TM-Derivative(+-).h5', 'w')
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'w')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_Inter-100_FeatSize-Variable_SigmaThreshol-None.h5', 'w')
    
    kfolds = 10
    vocabilary_size = [100000] #[1000,3000,10000,100000]
    iter_l = [100]
    featr_size_lst = [1000, 5000, 10000, 20000, 50000, 70000] 
    sigma_threshold = 0.8
    
    crossV_Koppels = CrossVal_Koppels_method(sparse_W, CrossVal_Kopples_method_res, corpus_filepath, genres)
    
    xhtml_file_l, cls_gnr_tgs = crossV_Koppels.corpus_files_and_tags()
    
    crossV_Koppels.evaluate(xhtml_file_l, cls_gnr_tgs, kfolds, vocabilary_size, iter_l, featr_size_lst,\
                                     sigma_threshold, similarity_func=cosine_similarity, norm_func=None)
    
    CrossVal_Kopples_method_res.close()

    
    

