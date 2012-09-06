"""

"""

import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
import numpy as np
import tables as tb
#import html2tf.tables.cngrams as cng_tb

import scipy.sparse as ssp
import scipy.spatial.distance as spd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn import cross_validation

import html2vect.sparse.words as h2v_w
import html2vect.sparse.cngrams as h2v_cng


        
class CrossVal_Mahal_method(object):
    
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
    
     
    def contruct_classes_mean_cov(self, trn_idxs, corpus_mtrx, cls_gnr_tgs, features_subspace):
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
    
        gnr_classes_icov = dict()
        gnr_classes_mean = dict()
        for g in self.genres_lst:
            #
            subspace_subcorpus = corpus_mtrx[inds_per_gnr[g], :].toarray()
            subspace_subcorpus = subspace_subcorpus[:, features_subspace]
            cov = np.cov(subspace_subcorpus, rowvar=0)
            print cov.shape()
            gnr_classes_icov[g] = np.linalg.inv( cov )
            print gnr_classes_icov[g]
            print
            gnr_classes_mean[g] = np.mean( subspace_subcorpus, axis=0) 
        
        return (gnr_classes_icov, gnr_classes_mean)   
    
    
    def predict(self, trn_idxs, corpus_mtrx, cls_gnr_tgs, crossval_X, crossval_Y, vocab_index_dct, featrs_size, similarity_func, sim_min_value, iters, sigma_threshold):
            
        max_sim_scores_per_iter = np.zeros((iters, crossval_X.shape[0]))
        predicted_classes_per_iter = np.zeros((iters, crossval_X.shape[0]))
                    
        #Measure similarity for iters iterations i.e. for iters different feature subspaces Randomly selected 
        for I in range(iters):
            
            #Randomly select some of the available features
            suffled_vocabilary_idxs = np.random.permutation( np.array(vocab_index_dct.values()) ) 
            features_subspace = suffled_vocabilary_idxs[0:featrs_size]
            
            print "Construct classes"
            #Construct Genres Class Vectors form Training Set
            #gnr_classes, inds_per_gnr = self.contruct_classes(trn_idxs, corpus_mtrx[0], cls_gnr_tgs)
            gnr_classes_icov, gnr_classes_mean = self.contruct_classes_mean_cov(trn_idxs, corpus_mtrx, cls_gnr_tgs, features_subspace)
            
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
                    sim_score = similarity_func(vect, gnr_classes_icov[g], gnr_classes_mean[g])
                    
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
        
    
    def evaluate(self, xhtml_file_l, cls_gnr_tgs, kfolds, vocabilary_size, iter_l, featr_size_lst, sigma_threshold, similarity_func, sim_min_val, norm_func):
        
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
                        predicted_classes_per_iter = self.predict(trn_idxs, corpus_mtrx[0], cls_gnr_tgs,\
                                                                  crossval_X, crossval_Y,\
                                                                  tid, featrs_size,\
                                                                  similarity_func, sim_min_val,\
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
                                        
            
def mahalanobis_similarity(vector, icov, mean):
    
    print "Mahalanobis"    
    return 1 / ( 1 + spd.mahalanobis(vector.todense(), mean, icov) ) 



if __name__ == '__main__':
    
    sparse_W = h2v_w.Html2TF(attrib='text', lowercase=True, valid_html=False)
    sparse_CNG = h2v_cng.Html2TF(4, attrib='text', lowercase=True, valid_html=False)
    
    corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
    #corpus_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/"
    genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    #genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop" ]
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santini_TT-Words_TM-Derivative(+-).h5', 'w')
    CrossVal_Mahal_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_Mahala.h5', 'w')
    # = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_Inter-100_FeatSize-Variable_SigmaThreshol-None.h5', 'w')
    
    kfolds = 10
    vocabilary_size = [10000] #[1000,3000,10000,100000]
    iter_l = [100]
    featr_size_lst = [1000] #, 5000, 10000, 20000, 50000, 70000] 
    sigma_threshold = 0.8
    
    crossV_Mahal = CrossVal_Mahal_method(sparse_W, CrossVal_Mahal_method_res, corpus_filepath, genres)
    
    xhtml_file_l, cls_gnr_tgs = crossV_Mahal.corpus_files_and_tags()
    
    crossV_Mahal.evaluate(xhtml_file_l, cls_gnr_tgs, kfolds, vocabilary_size, iter_l, featr_size_lst,\
                                     sigma_threshold, similarity_func=mahalanobis_similarity, sim_min_val=0.0, norm_func=None)
    
    CrossVal_Mahal_method_res.close()

    
    
    
    
    
    
    
    
    
    
    

