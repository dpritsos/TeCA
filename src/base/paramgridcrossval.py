

import json
import os
import cPickle as pickle

import numpy as np
import tables as tb

import scipy.sparse as ssp
import scipy.spatial.distance as spd

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn import cross_validation
from sklearn import grid_search



class ParamGridCrossValBase(object):
    
    def __init__(self, ML_Model, TF_TT, h5_res, genres, corpus_path, voc_path):
        self.model = ML_Model
        self.TF_TT = TF_TT
        self.corpus_path = corpus_path
        self.genres_lst = genres
        self.gnrs_num = len(genres)
        self.h5_res = h5_res
        self.crps_voc_path = voc_path


    def corpus_files_and_tags(self, gnr_file_idx=None, iidx=None):

        corpus_files_lst_path = self.crps_voc_path+'/Corpus_filename_shorted.lst'
        corpus_tags_lst_path = self.crps_voc_path+'/Corpus_tags_shorted.lst'

        if os.path.exists(corpus_files_lst_path) and  os.path.exists(corpus_files_lst_path):
            
            print "Loading HTML Filenames and Classes Tags Lists" 
            
            #Load Filename and classes Tags lists
            with open(corpus_files_lst_path, 'r') as f:
                html_file_l = json.load(f, encoding='utf-8')

            with open(corpus_tags_lst_path, 'r') as f:
                cls_gnr_tgs = json.load(f, encoding='utf-8')
            
        else:
            
            print "Creating and Saving HTML Filenames and Classes Tags Lists"
            
            html_file_l = list()
            cls_gnr_tgs = list()
            if gnr_file_idx and iidx:
                #Get the index file showing the respective genre of each file of the given corpus
                with open(self.corpus_path+gnr_file_idx, 'r') as f:
                    #Getting the list of tuples (index, file) using the index-of-index list (iidx) argument
                    #iidx[0] == the splittig character, iidx[1] == the field that contains the filename , 
                    #iidx[2] == the field that containd the genre of the file
                    gnrs_file_lst = [ (line.split( iidx[0] )[ iidx[2] ], line.split( iidx[0] )[ iidx[1] ]) for line.split( iidx[0] ) in f ]
                
                #Sort the above tuples list based on genre 
                sorted(gnrs_file_lst, key=lambda gnrs_file_lst: gnrs_file_lst[0])
                
                #just for debugging remove it right after
                print gnrs_file_lst

                #Get the filenames as sorted above
                html_file_l = [ element[1] for element in gnrs_file_lst ]
                
                #Build the class-genre-tag list by assigning as a tag the index number of the list of genre given as argument to this
                #class, i.e. ParamGridCrossValBase()
                cls_gnr_tgs = [ genres_lst.index( element[0] ) for element[1] in gnrs_file_lst ]

            elif gnr_file_idx == None  and iidx == None:
                #Get the list of Genre argument as given to this Class and build html-file-list and class-genres-tags list
                for i, g in enumerate(self.genres_lst):
                    #Get all files located to the genre's path 'g'
                    gnrs_file_lst = self.TF_TT.file_list_frmpaths(self.corpus_path, [ str( g + "/html/" ) ] )
                    
                    #Extends the list of html files with the set of files form genre 'g'
                    html_file_l.extend( gnrs_file_lst )
                    
                    #Extends the list of html files with the set of class tag form genre 'g', i.e. the index of the 
                    #genre's list given as argument to this class ( ParamGridCrossValBase() ).
                    cls_gnr_tgs.extend( [i+1]*len(gnrs_file_lst) )

            else:
                raise Exception("Both Genre-of-Files-Index and Index-of-Index arguments should be given or 'None' of them")

            #Saving Filename and classes Tags lists
            with open(corpus_files_lst_path, 'w') as f:
                #HTML File List as founded in the Ext4 file system by python built-it os (python 2.7.x) lib
                json.dump(html_file_l, f, encoding='utf-8')

            with open(corpus_tags_lst_path, 'w') as f:
                #Assigned Genre Tags to files list Array
                json.dump(cls_gnr_tgs, f, encoding='utf-8')
    
        return (np.array(html_file_l), np.array(cls_gnr_tgs))


    def calculate_p_r_f1(self, crossval_Y, predicted_Y):

        #Calculating Scores Precision, Recall and F1 Statistic
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

        return (P_per_gnr, R_per_gnr, F1_per_gnr)
    

    def evaluate(self, *args):
        """ Call prototyping: evaluate(html_file_l, cls_gnr_tgs, None, params_range, 'utf-8') """

        html_file_l = args[0]
        cls_gnr_tgs = args[1]
        norm_func = args[2]
        params_range = args[3]
        encoding = args[4]

        #Create CrossVal Folds
        KF = cross_validation.StratifiedKFold(cls_gnr_tgs, len(params_range['kfolds']), indices=True)
        
        for k, (trn, crv) in enumerate(KF):

            voc_filename = self.crps_voc_path+'/kfold_Voc_'+str(k)+'.vtf'
            pkl_voc_filename = self.crps_voc_path+'/kfold_Voc_'+str(k)+'.pkl'
            trn_filename = self.crps_voc_path+'/kfold_trn_'+str(k)+'.idx'
            crv_filename = self.crps_voc_path+'/kfold_crv_'+str(k)+'.idx'

            #Save K-Fold Cross-Validation corpus vector selection-indecies if does not exists
            #Create K-fold Cross-Validation Vocabulary for each fold
            #Stratified Indecies and respective Vocabularies should be syncornisied therefore 
            #there saving-files should be created all together if one or more are missing
            if not os.path.exists(trn_filename) or not os.path.exists(crv_filename) or not os.path.exists(voc_filename) or not os.path.exists(pkl_voc_filename):
                
                #Save Trainging Indeces
                print "Saving Training Indices for k-fold=", k
                with open(trn_filename, 'w') as f:
                    json.dump( list(trn), f, encoding=encoding)

                #Save Cross-validation Indeces
                print "Saving Cross-validation Indices for k-fold=", k
                with open(crv_filename, 'w') as f:
                    json.dump( list(crv), f, encoding=encoding)               
         
                #Creating Vocabulary
                print "Creating Vocabulary for k-fold=",k
                tf_d = self.TF_TT.build_vocabulary( list( html_file_l[trn] ), encoding='utf-8', error_handling='replace' )

                #Saving Vocabulary
                print "Saving Vocabulary"
                with open(pkl_voc_filename, 'w') as f:
                    pickle.dump(tf_d, f)

                with open(voc_filename, 'w') as f:
                    json.dump(tf_d, f, encoding=encoding)

        #Starting Parameters Grid Search 
        for params in grid_search.IterGrid(params_range):

            #Prevent execution of this loop in case feature_size is smaller than Vocabulary size
            if params['features_size'] > params['vocab_size']:
                print "Skipped Params: ", params
                continue                    

            print "Params: ", params
            #bs = cross_validation.Bootstrap(9, random_state=0)
            #Set Experiment Parameters
            k = params['kfolds']
            vocab_size = params['vocab_size']
            featrs_size = params['features_size']

            #Creating a Group for this Vocabulary size in h5 file under this k-fold
            try:
                vocab_size_group = self.h5_res.getNode('/', 'vocab_size'+str(vocab_size))    
            except:
                vocab_size_group = self.h5_res.createGroup('/', 'vocab_size'+str(vocab_size),\
                                "Vocabulary actual size group of Results Arrays for this K-fold" )

            #Creating a Group for this features size in h5 file under this k-fold
            try:
                feat_num_group = self.h5_res.getNode(vocab_size_group, 'features_size'+str(featrs_size))    
            except:
                feat_num_group = self.h5_res.createGroup(vocab_size_group, 'features_size'+str(featrs_size),\
                                "Features Number group of Results Arrays for this K-fold" )
            
            ###Create the group sequence respectively to the models parameters:

            #Assigne Feature number group to next_group parameter for initializing the loop
            next_group = feat_num_group

            #Start the loop of creating of getting group nodes in respect to model parameters
            for pname, pvalue in params.items():
                if pname not in ['kfolds', 'vocab_size', 'features_size']:           
                    try:
                        next_group = self.h5_res.getNode(next_group, pname+str(pvalue).replace('.',''))
                    except:
                        next_group = self.h5_res.createGroup(next_group, pname+str(pvalue).replace('.',''), "<Comment>" )   

            ###END- Group sequence creation

            #Creating a Group for this k-fold in h5 file
            try:
                kfld_group = self.h5_res.getNode(next_group, 'KFold'+str(k))
            except:
                kfld_group = self.h5_res.createGroup(next_group, 'KFold'+str(k), "K-Fold group of Results Arrays")

            #Loading Vocavulary
            print "Loadinging VOCABULARY for k-fold=",k
            with open(pkl_voc_filename, 'r') as f:
                tf_d = pickle.load(f)
            
            #Get the Vocabuliary keeping all the terms with same freq to the last feature of the reqested size
            resized_tf_d = self.TF_TT.tfdtools.keep_atleast(tf_d, vocab_size) 

            #Create The Terms-Index Vocabulary that is shorted by Frequency descending order
            tid = self.TF_TT.tfdtools.tf2tidx( resized_tf_d )
            print tid.items()[0:5]

            #Load or Crreate the Coprus Matrix (Spase) for this combination or kfold and vocabulary_size
            corpus_mtrx_fname = self.crps_voc_path+'/kfold_CorpusMatrix_'+str(k)+str(vocab_size)+'.pkl'

            if os.path.exists(corpus_mtrx_fname):
                print "Loading Sparse TF Matrix for CrossValidation for K-fold=", k, " and Vocabulary size=", vocab_size
                #Loading Coprus Matrix (Spase) for this combination or kfold and vocabulary_size
                with open(corpus_mtrx_fname, 'r') as f:
                    corpus_mtrx = pickle.load(f)

            else:
                print "Creating Sparse TF Matrix (for CrossValidation) for K-fold=", k, " and Vocabulary size=", vocab_size
                #Creating TF Vectors Sparse Matrix
                corpus_mtrx = self.TF_TT.from_files(list( html_file_l ), tid_dictionary=tid, norm_func=norm_func,\
                                                    encoding='utf8', error_handling='replace' )[0]

                #Saving TF Vecrors Matrix
                print "Saving Sparse TF Matrix (for CrossValidation)"
                with open(corpus_mtrx_fname, 'w') as f:
                    pickle.dump(corpus_mtrx, f)

            #Save Vocabulary and Documents Sizes for this experiment (i.e. this kfold, this text representation, etc.)
            #Save them for an other fold only if the Vocabulary size is different (Most likely same-sized Vocabularies means identical ones)
            #
            #For the first k-fold just save them all 
            if k == 0:
                #keep as pytables group attribute the actual Vocabulary size
                vocab_size_group._v_attrs.real_voc_size = [(k, len(resized_tf_d))]
                
                #Save the Webpages term counts (Char N-grans or Word N-Grams)
                docs_term_counts = self.h5_res.createArray(vocab_size_group, 'docs_term_counts', corpus_mtrx.sum(axis=1))

            else:
                #For the rest of k-folds save the current Vocabulary and the Corpus Documents sizes if current Vocabulary size is different to the previous ones
                if len(resized_tf_d) not in vocab_size_group._v_attrs.real_voc_size:
                    
                    #Add the Vocabulary size on the list of real Vocabilary sized for each fold
                    vocab_size_group._v_attrs.real_voc_size += [(k,len(resized_tf_d))]

                    #If Vocabulary is different the the Document term counts will be differnet, then save them again
                    docs_term_counts = self.h5_res.createArray(vocab_size_group, 'docs_term_counts'+str(k), corpus_mtrx.sum(axis=1))

            #Perform default (division by max value) normalisation for corpus matrix 'corpus_mtrx'
            #Should I perform Standarisation/Normalisation by substracting mean value from vector variables?
            corpus_mtrx = ssp.csr_matrix( corpus_mtrx.todense() / np.max(corpus_mtrx.todense(), axis=1) )
                
            #Load Training Indeces 
            trn_filename = self.crps_voc_path+'/kfold_trn_'+str(k)+'.idx'
            print "Loading Training Indices for k-fold=", k
            with open(trn_filename, 'r') as f:
                trn_idxs = np.array( json.load(f, encoding=encoding) )

            #Load Cross-validation Indeces
            crv_filename = self.crps_voc_path+'/kfold_crv_'+str(k)+'.idx'
            print "Loading Cross-validation Indices for k-fold=", k
            with open(crv_filename, 'r') as f:
                crv_idxs = np.array( json.load(f, encoding=encoding) )

            #Select Cross Validation Set
            #crossval_Y = cls_gnr_tgs[ crv_idxs ]
            #mtrx = corpus_mtrx
            #crossval_X = mtrx[crv_idxs, :]

            print "EVALUATE"
            #Evaluating Classification Method
            predicted_Y, predicted_scores,\
            model_specific_d = self.model.eval(\
                                    trn_idxs, crv_idxs,\
                                    corpus_mtrx, cls_gnr_tgs, tid,\
                                    params\
                                ) 

            #Select Cross Validation Set
            crossval_Y = cls_gnr_tgs[ crv_idxs ]
            #mtrx = corpus_mtrx
            #crossval_X = mtrx[crv_idxs, :]
                
            P_per_gnr, R_per_gnr, F1_per_gnr = self.calculate_p_r_f1(crossval_Y, predicted_Y)
                        
            #Saving results
            print self.h5_res.createArray(kfld_group, 'expected_Y', crossval_Y, "Expected Classes per Document (CrossValidation Set)")[:]                                         
            print self.h5_res.createArray(kfld_group, 'predicted_Y', predicted_Y, "predicted Classes per Document (CrossValidation Set)")[:]
            print self.h5_res.createArray(kfld_group, 'predicted_scores', predicted_scores, "predicted Scores per Document (CrossValidation Set)")[:]
            print self.h5_res.createArray(kfld_group, "P_per_gnr", P_per_gnr, "Precision per Genre (P[0]==Global P)")[:]
            print self.h5_res.createArray(kfld_group, "R_per_gnr", R_per_gnr, "Recall per Genre (R[0]==Global R)")[:]
            print self.h5_res.createArray(kfld_group, "F1_per_gnr", F1_per_gnr, "F1_statistic per Genre (F1[0]==Global F1)")[:]
            
            for name, value in model_specific_d.items():
                print self.h5_res.createArray(kfld_group, name, value, "<Comment>")[:]             
            
        #Return Resuls H5 File handler class
        return self.h5_res                                    

 