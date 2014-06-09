""" 
    This module includes OC-SVME data retrieval functions.
    Details can be found in each funciton separatly. 

    Aurthor: Dimitrios Pritos

"""

import sys
sys.path.append('../../src')

import numpy as np


def get_predictions(res_h5file, kfolds, params_path, genre_tag=None):

    """Retrieval functions for the date returned from the OC-SVME method. 

    Returns the Predicted Scores and the Expected Values for the sample set has been given to the 
    RSFE for evaluation. If genre_tag argument is a number other that None then it returns the perdition
    scores and expected values for the given genre tag. In case of None value, it returns the Truth Table 
    of Predicted and Expected values of all genre in the sample. 

    The default case genre_tag (i.e. == None) is base on the concept that a Precision Recall Carve is the fraction 
    of true positives (TP) in relation to all positive predictions (i.e. known as precision). Therefore, one can use
    The Truth Table is created in Default case for evaluating the OC-SVME for the whole Sample. In particular using
    PRC for evaluation, it will be shown that rate of the Successful Perdition (i.e. predicted as positive while 
    they where expected as positive) compare to all the positive (TP + FP) predictions. However, using this for ROC
    curves things are a bit more complicated therefore I recommend not to use this argument (genre_tag=None) 
    for ROC curves. 

   Input argumens:
        
        res_h5file: The HD5 file object where the results are expected to be located.

        kfolds: A list of fold numbers will be retrieved and averaged. 

        params_path: The path in HD5 where the Expected/Predicted Y and Score tables are located.
            The HD5 file is expected to have a path where in each node there is a parameter value used for the 
            experiments.

        genre_tag: The number (integer) which is the tag of the genre to be retrieved. 
            Default value: None. In this case the EY will contain the Truth Table of Expected and Predicted values.

    Output:

        PS: The predicted Scores of the OC-SVME for each sample in the corpus given for evaluation to the OC-SVME method. 
        EY: The expected values in binary format {0,1}. However in that case of genre_tag=None it will return the 
            Truth Table of Expected and Predicted values.

    """

    #Initialising.
    PS_lst = list() #Ensemble Predicted Scores
    EY_lst = list() #Ensemble Truth Table
    PR_Y_lst = list() #Ensemble Predicted Y's
    

    #Choosing whether or not to create a Truth table depeding on the genre_tag value.
    if isinstance(genre_tag, int):

        #Collecting Scores for and Expected Values for every fold given in kfold list.
        for k in kfolds:

            raise Exception("It is not implemented yet!")

            """

            #Calulating Prediction Scores for the given Gerne i.e. assuming that the rest genres being Negative examples
            pc_per_iter = res_h5file.get_node(params_path+'/KFold'+str(k), name='predicted_classes_per_iter').read()
            #pc_per_iter = pc_per_iter[0:-2500]
            #genre_tag = genre_tag[0:-2500]
            gnr_pred_cnt = np.where(pc_per_iter == genre_tag, 1, 0) 
            
            fold_ps = np.sum(gnr_pred_cnt, axis=0) / np.float(pc_per_iter.shape[0])   
            PS_lst.append(fold_ps)
            
            #Collecting the excected tag values by conveting them first in binary form.
            exp_y = res_h5file.get_node(params_path+'/KFold'+str(k), name='expected_Y' ).read()
            #exp_y = exp_y[0:-2500]
            EY_lst.append( np.where(exp_y == genre_tag, 1, 0) ) 

            """

    elif genre_tag == None:

        #Collecting Scores for and Expected Values for every fold given in kfold list.    
        for k in kfolds:
            
            #Loading expected and predicted values.
            pred_scores = res_h5file.get_node(params_path+'/KFold'+str(k), name='predicted_scores').read()
            pred_scores = pred_scores#[0:-1000]
            PS_lst.append( pred_scores )

            exp_y = res_h5file.get_node(params_path+'/KFold'+str(k), name='expected_Y').read()
            exp_y = exp_y#[0:-1000]

            pre_y = res_h5file.get_node(params_path+'/KFold'+str(k), name='predicted_Y').read()
            pre_y = pre_y#[0:-1000]

            PR_Y_lst.append(pre_y)

            #Collecting and the Truth Table of expected and predicted values.
            EY_lst.append( np.where(exp_y == pre_y, 1, 0) )

    else:
        raise Exception("Invalid genre_tag argument's value: Valid arguments are either and integer (as genre tag) of 'None'")


    #Stack the lists to Single arrays for PS and EY respectively
    PS = np.hstack(PS_lst)
    EY = np.hstack(EY_lst)
    PR_Y = np.hstack(PR_Y_lst)

    #print PR_Y[ PR_Y == 0 ]
    
    #Short Results by Predicted Scores
    inv_srd_idx = np.argsort(PS)[::-1]
    PS = PS[ inv_srd_idx ]
    EY = EY[ inv_srd_idx ]
    PR_Y = PR_Y[inv_srd_idx]

    #Retunring Predicted Scores and Expected Values
    return (PS, EY, PR_Y)



