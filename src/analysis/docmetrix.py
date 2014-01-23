
import json
import pickle as pkl
import numpy as np

from sklearn import grid_search



def get_idx2gnr(root_path, corpus_lst_fname):

    with open(root_path + corpus_lst_fname, 'r')  as f:
        flst = json.load(f)
        gnr_lst = [ fpth.split('/')[5] for fpth in flst ]

    return gnr_lst


def get_docsizes(res_h5d_f, voc_size, kfolds=0):
    """Gets the Documents Sizes Array been stored in results H5D file

        Argumens: 
            res_h5d_f: H5D file contains results from AGI experiments.
            voc_size: 
            kfolds***: The number of kfolds performed in corssvalidation.

        Returns:
            docs_sizes_arr_list: A list of numpy.array contains the 
            terms counts of all files in corpus.
    """

    docs_sizes_arr_lst = list()

    for kf in kfolds:
        docs_sizes = res_h5d_f.getNode('/vocab_size'+str(voc_size)+'/KFold'+str(kf), name='docs_term_counts' ).read()
        docs_sizes_arr_lst.append( docs_sizes )

    return docs_sizes_arr_lst


def Docs_Sizes(res_h5file, kfolds, params_path, root_path, fidx2gnr=None):

    #Listed on:
    # Exepcted Y
    # Predicted Y
    # Predicted Score
    
    EY_lst = list()
    PY_lst = list()
    DCL_lst = list()
    PS_lst = list()
    crv_idxs_lst = list()

    prmpath = '/' + params_path.split('/')[1] 
    doc_len = res_h5file.getNode(prmpath+'/KFold'+str(0), name='docs_term_counts' ).read()
    DCL_lst.append( doc_len[0::] )
    
    for k in kfolds:

        #pc_per_iter = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_classes_per_iter').read()
        #gnr_pred_cnt = np.where(pc_per_iter == genre_tag, 1, 0) 
        #fold_ps = np.sum(gnr_pred_cnt, axis=0) / np.float(pc_per_iter.shape[0])   
        #PS_lst.append(fold_ps) 

        #Loading Expected_Y indices for this fold --- temp 0 fold
        with open(root_path + 'kfold_crv_' + str(k) + '.idx', 'r')  as f:
            crv_idxs = json.load(f)
        crv_idxs_lst.append(crv_idxs)
        
        #pred_scores = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_scores' ).read()
        #PS_lst.append( pred_scores )
        
        exp_y = res_h5file.getNode(params_path+'/KFold'+str(k), name='expected_Y' ).read()
        EY_lst.append( exp_y ) #[0:-1000] )

        pre_y = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_Y' ).read()
        PY_lst.append( pre_y ) #[0:-1000] )

    #EY_lst.append( exp_y[1000::] )
    #PY_lst.append( pre_y[1000::] )

    #Make Tables for Ensemble Algo
    EY = np.hstack(EY_lst)
    PY = np.hstack(PY_lst)
    DCL = np.hstack(DCL_lst)
    CRV = np.hstack(crv_idxs_lst)
    #PS = np.hstack(PS_lst)

    #print DCL.shape

    #Short Results by Cross Validation Idices
    crv_idx_idx = np.argsort(CRV)
    
    #Short Results by Predicted Scores
    #inv_srd_idx = np.argsort(EY)[:]
    PY = PY[ crv_idx_idx ]
    EY = EY[ crv_idx_idx ]
    CRV = CRV[ crv_idx_idx ]

    #CVR = CRV[ crv_idxs_idx ]

    #print PY

    Y = DCL # [ (PY == EY ) & (PY == 0) ]

    print len(EY == 0)
    constrain_inds = CRV[ (PY != EY)]

    print constrain_inds
    
    if fidx2gnr:

        #Cutting X_len in Genre Sets
        Y_sets = list()
        y_set = list()
        X_sets = list()

        last_gnr = fidx2gnr[0]
        Z_sets = list()
        Z_sets.append(last_gnr)

        for crv_i in CRV:
            
            gname = fidx2gnr[crv_i]

            if gname != last_gnr: 
                last_gnr = gname 
                Z_sets.append(last_gnr)
                Y_sets.append(y_set)    
                X_sets.append( range(len(y_set)) )
                print len(y_set)
                y_set = []
           
            if crv_i in constrain_inds:
                y_set.append( Y[crv_i] )

            else:
                pass #y_set.append( -10 )

        Y_sets.append(y_set)   
        print len(y_set) 
        X_sets.append( range(len(y_set)) )
        print

        Y = Y_sets
        X = X_sets
        Z = Z_sets

    else:
        X = range( Y.shape[0] )
        Z = [0]

    return X, Y, Z
