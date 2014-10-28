"""
    This module includes RFSE data retrieval functions.
    Details can be found in each funciton separatly.

    Aurthor: Dimitrios Pritos

"""

import sys
sys.path.append('../../src')

import numpy as np


def get_predictions(hf5_fl1, hf5_fl2, kfolds, params_path, sigma, gnr_num,
                    genre_tag=None, binary=None):

    """Retrieval functions for the date returned from the RFSE method.

    Returns the Predicted Scores and the Expected Values for the sample set has been given to the
    RSFE for evaluation. If genre_tag argument is a number other that None then it returns the
    perdition scores and expected values for the given genre tag. In case of None value, it returns
    the Truth Table of Predicted and Expected values of all genre in the sample.

    The default case genre_tag (i.e. == None) is base on the concept that a Precision Recall Carve
    is the fraction of true positives (TP) in relation to all positive predictions
    (i.e. known as precision). Therefore, one can use The Truth Table is created in Default case
    for evaluating the RFSE for the whole Sample. In particular using PRC for evaluation, it will
    be shown that rate of the Successful Perdition (i.e. predicted as positive while they where
    expected as positive) compare to all the positive (TP + FP) predictions. However, using this
    for ROC curves things are a bit more complicated therefore I recommend not to use this argument
    (genre_tag=None) for ROC curves.

   Input argumens:

        ????????????????????????hf5_fl1: The HD5 file object where the results are expected to be located.????????????????

        kfolds: A list of fold numbers will be retrieved and averaged.

        params_path: The path in HD5 where the Expected/Predicted Y and Score tables are located.
            The HD5 file is expected to have a path where in each node there is a parameter value
            used for the experiments.

        genre_tag: The number (integer) which is the tag of the genre to be retrieved.
            Default value: None. In this case the EY will contain the Truth Table of Expected and
            Predicted values.

    Output:

        PS: The predicted Scores of the RFSE for each sample in the corpus given for evaluation to
        the RFSE method.
        EY: The expected values in binary format {0,1}. However in that case of genre_tag=None it
        will return the Truth Table of Expected and Predicted values.

    """

    #Initialising.
    #Ensemble Predicted Scores
    PS_lst = list()
    #Ensemble Truth Table
    EY_lst = list()
    #Ensemble Predicted Y's
    PR_Y_lst = list()


    #Choosing whether or not to create a Truth table depeding on the genre_tag value.
    if genre_tag == None:

        #Collecting Scores for and Expected Values for every fold given in kfold list.
        for k in kfolds:

            #Calulating Prediction Scores for the given Gerne i.e. assuming that the rest genres
            #being Negative examples

            #File #1
            pc_array_fl1 = hf5_fl1.get_node(
                params_path + '/KFold'+str(k), name='predicted_classes_per_iter'
            ).read()

            ms_array_fl1 = hf5_fl1.get_node(
                params_path+'/KFold'+str(k), name='max_sim_scores_per_iter'
            ).read()

            #File #2
            pc_array_fl2 = hf5_fl2.get_node(
                params_path + '/KFold'+str(k), name='predicted_classes_per_iter'
            ).read()

            ms_array_fl2 = hf5_fl2.get_node(
                params_path+'/KFold'+str(k), name='max_sim_scores_per_iter'
            ).read()

            #Normalising max-similarity scores for being comparable irrespectively of the
            #measure has been used.
            max_ms = np.max(ms_array_fl1, axis=0)
            min_ms = np.min(ms_array_fl1, axis=0)
            ms_array_fl1 = (ms_array_fl1 - min_ms) / (max_ms - min_ms)

            max_ms = np.max(ms_array_fl2, axis=0)
            min_ms = np.min(ms_array_fl2, axis=0)
            ms_array_fl2 = (ms_array_fl2 - min_ms) / (max_ms - min_ms)

            docs_num = ms_array_fl1.shape[1]
            itrs = np.arange(ms_array_fl1.shape[0])
            ms_arr1_cls = np.hsplit(ms_array_fl1, docs_num)
            ms_arr2_cls = np.hsplit(ms_array_fl2, docs_num)
            pc_arr1_cls = np.hsplit(pc_array_fl1, docs_num)
            pc_arr2_cls = np.hsplit(pc_array_fl2, docs_num)

            #print pc_arr2_cls

            new_pc_col_lst = list()

            for i, (ms_c1, ms_c2, pc_c1, pc_c2) in enumerate(zip(
                    ms_arr1_cls, ms_arr2_cls, pc_arr1_cls, pc_arr2_cls)):

                i_max_ms_col = np.argmax(np.hstack((ms_c1, ms_c2)), axis=1)

                #EXPLAIN THIS IN DETAIL...
                pc4max_ms = np.hstack((pc_c1, pc_c2))[itrs, i_max_ms_col]

                new_pc_col_lst.append(pc4max_ms)

            #The matrix of predicted classes per iteration. It is transposed because vstack is
            #required for lists to be concatenated correctly.
            ###pc_array = np.vstack(new_pc_col_lst).T

            #Calculating the Predicted Class and the Sigma scores for each document.
            pre_y = np.zeros(docs_num)
            pre_score = np.zeros(docs_num)

            for i, pc_col in enumerate(new_pc_col_lst):

                max_gnr_scr = np.bincount(pc_col.astype(np.int), minlength=8) / float(len(itrs))

                if sigma < np.max(max_gnr_scr):
                    pre_y[i] = np.argmax(max_gnr_scr)
                    pre_score[i] = np.max(max_gnr_scr)

            #Saving the Ensemble Prediction Scores.
            PS_lst.append(pre_score)

            #Saving the Class predictions.
            PR_Y_lst.append(pre_y)

            #Getting the Expected genre tags
            exp_y = hf5_fl1.get_node(params_path+'/KFold'+str(k), name='expected_Y').read()

            #Collecting and the Truth Table of expected and predicted values.
            EY_lst.append(exp_y)

    else:
        raise Exception(
            "Invalid genre_tag argument's value: Valid arguments are either" +
            " and integer (as genre tag) of 'None'"
        )

    """
    elif isinstance(genre_tag, int):

    """

    #Stacking the lists to Single arrays for PS and EY respectively
    PS = np.hstack(PS_lst)
    EY = np.hstack(EY_lst)
    PR_Y = np.hstack(PR_Y_lst)

    #Converting in to binary case under binary variable condition.
    if binary:
        EY = np.where(EY == PR_Y, 1, 0)

    #Shorting Results by Predicted Scores
    inv_srd_idx = np.argsort(PS)[::-1]
    PS = PS[inv_srd_idx]
    EY = EY[inv_srd_idx]
    PR_Y = PR_Y[inv_srd_idx]

    #Retunring Predicted Scores and Expected Values
    return (PS, EY, PR_Y)
