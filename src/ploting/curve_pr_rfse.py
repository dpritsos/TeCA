
"Precision Recall Curves"

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import pr_curves_to_11_standard_recall_levels as srl

from sklearn import grid_search


def prcurve(res_h5file, kfolds, params_path, genre_tag=None, mark_thres=None):

    PS_lst = list() #Ensemble Algo Predicted Scores
    EY_lst = list() #Ensemble Algo Truth Table
    
    if genre_tag:

        for k in kfolds:

            pc_per_iter = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_classes_per_iter').read()
            gnr_pred_cnt = np.where(pc_per_iter == genre_tag, 1, 0) 
            
            fold_ps = np.sum(gnr_pred_cnt, axis=0) / np.float(pc_per_iter.shape[0])   
            PS_lst.append(fold_ps) 
            
            exp_y = res_h5file.getNode(params_path+'/KFold'+str(k), name='expected_Y' ).read()
            EY_lst.append( np.where(exp_y == genre_tag, 1, 0) ) 

    else:    

        for k in kfolds:
            
            pred_scores = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_scores').read()
            PS_lst.append( pred_scores )

            exp_y = res_h5file.getNode(params_path+'/KFold'+str(k), name='expected_Y').read()

            pre_y = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_Y').read()

            #Use the Truth Table for converting the exp_y to Binary case and append for this fold    
            EY_lst.append( np.where(exp_y == pre_y, 1, 0) ) 

    #Make Tables for Ensemble Algo
    PS = np.hstack(PS_lst)
    EY = np.hstack(EY_lst)
    
    #Short Results by Predicted Scores
    inv_srd_idx = np.argsort(PS)[::-1]
    PS = PS[ inv_srd_idx ]
    EY = EY[ inv_srd_idx ]

    #Calculate P-R Curves for Ensemble Algorithm
    P, R, T= skm.precision_recall_curve(EY, PS)
    
    T = T[::-1]
    R = R[::-1]
    P = P[::-1]

    if mark_thres != None:
        mark = np.max(np.where(T >= mark_thres))
        mark_X = R[mark]
        mark_Y = P[mark]
    else: 
        mark_X = None
        mark_Y = None
    
    #x, y = srl.std_avg_pr_curve(P[::-1], R[::-1])
    #X = R

    Y, X = srl.interpol_soothed_pr_curve(P[::-1], R)
    
    return (X, Y, mark_X, mark_Y)



def zero_class_dist(res_h5file, kfolds, params_path, genre_tag=None):
    
    PS_lst = list()
    PY_lst = list()

    if genre_tag:

        for k in kfolds:

            pc_per_iter = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_classes_per_iter').read()
            gnr_pred_cnt = np.where(pc_per_iter == genre_tag, 1, 0) 
            
            fold_ps = np.sum(gnr_pred_cnt, axis=0) / np.float(pc_per_iter.shape[0])   
            PS_lst.append(fold_ps) 

            pre_y = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_Y' ).read()

            PY_lst.append(pre_y)

    else:

        for k in kfolds:
            
            pred_scores = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_scores' ).read()
            PS_lst.append( pred_scores )

            pre_y = res_h5file.getNode(params_path+'/KFold'+str(k), name='predicted_Y' ).read()

            PY_lst.append(pre_y)

    #Make Tables for Ensemble Algo
    PS = np.hstack(PS_lst)
    PY = np.hstack(PY_lst)

    #Short Results by Predicted Scores
    inv_srd_idx = np.argsort(PS)[::-1]
    PY = PY[ inv_srd_idx ]
    
    Zero_Dist = np.where(PY == 0)

    return Zero_Dist
