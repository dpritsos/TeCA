

import numpy as np
import scipy.spatial.distance as spd



class RFSEBAGG_Wrapped(object):
    
    def __init__(self, similarity_func, sim_min_value, genres):
        self.similarity_func = similarity_func
        self.sim_min_value =  sim_min_value
        self.genres_lst = genres
        self.gnrs_num = len(genres)
        

    def contruct_classes(self, trn_idxs, corpus_mtrx, cls_gnr_tgs, bagging_param):
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
            
            #######
            shuffled_train_idxs = np.random.permutation( inds_per_gnr[g] )
            #print shuffled_train_idxs
            #keep bagging_parram percent
            bg_trn_ptg = int( np.trunc( shuffled_train_idxs.size * bagging_param ) )
            #print bg_trn_ptg
            bag_idxs = shuffled_train_idxs[0:bg_trn_ptg]
            #print bag_idxs
            ######
        
            #Merge All Term-Frequency Dictionaries created by the Raw Texts
            gnr_classes[g] = corpus_mtrx[bag_idxs, :].mean(axis=0)
        
        return gnr_classes

    
    def predict(self, *args):

        #Get Input arguments in given sequence
        trn_idxs = args[0]  
        crv_idxs = args[1]                             
        corpus_mtrx = args[2]
        cls_gnr_tgs = args[3]
        vocab_index_dct = args[4] #tid
        params = args[5]
        
        #Get the part of matrices and arrays required for the model predicition phase
        crossval_X =  corpus_mtrx[ crv_idxs ] 
        crossval_Y =  cls_gnr_tgs[ crv_idxs ]
        
        #Modle's parameters to be used in prediciton phase
        featrs_size =  params['features_size']
        iters =  params['Iterations']
        sigma_threshold = params['Sigma']
        bagging_param = params['Bagging']
            
        max_sim_scores_per_iter = np.zeros((iters, crossval_X.shape[0]))
        predicted_classes_per_iter = np.zeros((iters, crossval_X.shape[0]))
                    
        #Measure similarity for iters iterations i.e. for iters different feature subspaces Randomly selected 
        for I in range(iters):

            #print "Construct classes"
            #Construct Genres Class Vectors form Training Set
            gnr_classes = self.contruct_classes(trn_idxs, corpus_mtrx, cls_gnr_tgs, bagging_param)
            
            #Randomly select some of the available features
            shuffled_vocabilary_idxs = np.random.permutation( np.array(vocab_index_dct.values()) ) 
            features_subspace = shuffled_vocabilary_idxs[0:featrs_size]
            
            #Initialised Predicted Classes and Maximum Similarity Scores Array for this i iteration 
            predicted_classes = np.zeros( crossval_X.shape[0] )
            max_sim_scores = np.zeros( crossval_X.shape[0] )
            
            #Measure similarity for each Cross-Validation-Set vector to each available Genre Class(i.e. Class-Vector). For This feature_subspace
            for i_vect, vect in enumerate(crossval_X[:, features_subspace]):
                
                #Convert TF vectors to Binary 
                #vect_bin = np.where(vect[:, :].toarray() > 0, 1, 0) #NOTE: with np.where Always use A[:] > x instead of A > x in case of Sparse Matrices
                #print vect.shape
                
                max_sim = self.sim_min_value
                for cls_tag, g in enumerate(self.genres_lst):
                    
                    #Convert TF vectors to Binary
                    #gnr_cls_bin = np.where(gnr_classes[ g ][:, features_subspace] > 0, 1, 0)
                    #print gnr_cls_bin.shape
                    
                    #Measure Similarity
                    sim_score = self.similarity_func(vect, gnr_classes[ g ][:, features_subspace])
                    
                    #Just for debugging for 
                    #if sim_score < 0.0:
                    #    print "ERROR: Similarity score unexpected value ", sim_score
                    
                    #Assign the class tag this vector is most similar and keep the respective similarity score
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
            if np.max(genres_probs) >= sigma_threshold:
                predicted_Y[i_prd_cls] = np.argmax( genres_probs )
                predicted_scores[i_prd_cls] = np.max( genres_probs ) 

        model_specific_d = {\
            'max_sim_scores_per_iter' : max_sim_scores_per_iter,\
            'predicted_classes_per_iter' : predicted_classes_per_iter 
        }
        
        return predicted_Y, predicted_scores, model_specific_d
        
 

def cosine_similarity(vector, centroid):
 
    return vector * np.transpose(centroid) / ( np.linalg.norm(vector.todense()) * np.linalg.norm(centroid) )



def hamming_similarity(vector, centroid):
 
    return 1.0 - spd.hamming(centroid, vector)



def correlation_similarity(vector, centroid):
    
    vector = vector[0]
    centroid = np.array(centroid)[0]
        
    vector_ = np.where(vector > 0, 0, 1)
    centroid_ = np.where(centroid > 0, 0, 1)
   
    s11 = np.dot(vector, centroid)
    s00 = np.dot(vector_, centroid_)
    s01 = np.dot(vector_, centroid)
    s10 = np.dot(vector,centroid_)
    
    denom = np.sqrt((s10+s11)*(s01+s00)*(s11+s01)*(s00+s10))
    if denom == 0.0:
        denom = 1.0
        
    return (s11*s00 - s01*s10) / denom
