"""
    This module includes RFSE data retrieval functions.
    Details can be found in each funciton separatly.

    Aurthor: Dimitrios Pritos

"""

import sys
sys.path.append('../../src')

import numpy as np


def get_predictions(res_h5file, params_path, genre_tag=None, binary=None, strata=None):
    """Retrieval functions for the date returned from the RFSE method.

    Returns the Predicted Scores and the Expected Values for the sample set has been given to the
    RSFE for evaluation. If genre_tag argument is a number other that None then it returns the
    perdition scores and expected values for the given genre tag. In case of None value, it returns
    the Truth Table of Predicted and Expected values of all genre in the sample.

    The default case genre_tag (i.e. == None) is base on the concept that a Precision Recall Carve
    is the fraction of true positives (TP) in relation to all positive predictions (i.e. known as
    precision). Therefore, one can use The Truth Table is created in Default case for evaluating
    the RFSE for the whole Sample. In particular using PRC for evaluation, it will be shown that
    rate of the Successful Perdition (i.e. predicted as positive while they where expected as
    positive) compare to all the positive (TP + FP) predictions. However, using this for ROC curves
    things are a bit more complicated therefore I recommend not to use this argument
    (genre_tag=None) for ROC curves.

   Input argumens:

        res_h5file: The HD5 file object where the results are expected to be located.

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

    #  Initializing.
    PS_lst = list()  # Ensemble Predicted Scores
    EY_lst = list()  # Ensemble Truth Table
    PR_Y_lst = list()  # Ensemble Predicted Y's

    #  Choosing whether or not to create a Truth table depending on the genre_tag value.
    if isinstance(genre_tag, int):

        if strata:
            raise Exception("Strata argument not implemented to work in combination with genre_tag")

        #  Collecting Scores for and Expected Values for every fold given in kfold list.
        for k in kfolds:

            # Calculating Prediction Scores for the given Genre i.e. assuming that the
            # rest genres being Negative examples
            pc_per_iter = res_h5file.get_node(
                params_path + '/KFold' + str(k), name='predicted_classes_per_iter').read()
            # pc_per_iter = pc_per_iter[0:-2500]
            # genre_tag = genre_tag[0:-2500]
            gnr_pred_cnt = np.where(pc_per_iter == genre_tag, 1, 0)

            fold_ps = np.sum(gnr_pred_cnt, axis=0) / np.float(pc_per_iter.shape[0])
            PS_lst.append(fold_ps)

            #  Collecting the excepted tag values by conventing them first in binary form.
            exp_y = res_h5file.get_node(params_path + '/KFold' + str(k), name='expected_Y').read()
            # exp_y = exp_y[0:-2500]

            # # # # SOS - NEED TO BE TESTED AGAIN
            EY_lst.append(exp_y)

    elif genre_tag is None:

        #  Collecting Scores for and Expected Values for every fold given in kfold list.
        for k in kfolds:

            #  Loading predicted scores.
            pre_score = res_h5file.get_node(
                params_path + '/KFold' + str(k), name='predicted_scores'
            ).read()

            #  Getting the Expected genre tags.
            exp_y = res_h5file.get_node(params_path + '/KFold' + str(k), name='expected_Y').read()

            #  Getting the Expected genre tags.
            pre_y = res_h5file.get_node(params_path + '/KFold' + str(k), name='predicted_Y').read()

            #  Making a stratified selection upon a specific group of the results.
            if strata:

                gpr_idx = -strata[1]

                pre_score_grp1 = pre_score[0:gpr_idx]
                pre_score_grp2 = pre_score[gpr_idx::]

                pre_y_grp1 = pre_y[0:gpr_idx]
                pre_y_grp2 = pre_y[gpr_idx::]

                exp_y_grp1 = exp_y[0:gpr_idx]
                exp_y_grp2 = exp_y[gpr_idx::]

                #  Performing Stratified selection.
                unq_pred_tgs = np.unique(pre_y_grp2)

                strata_idxs = np.hstack(
                    [idx_arr[0:int(np.rint(idx_arr.shape[0]/strata[0]))] for idx_arr
                        in [np.where((pre_y_grp2 == tg))[0] for tg in unq_pred_tgs]]
                )

                pre_score = np.hstack((pre_score_grp1, pre_score_grp2[strata_idxs]))
                pre_y = np.hstack((pre_y_grp1, pre_y_grp2[strata_idxs]))
                exp_y = np.hstack((exp_y_grp1, exp_y_grp2[strata_idxs]))

            #  Saving the Ensemble Prediction Scores.
            PS_lst.append(pre_score)

            #  Saving the Class predictions.
            PR_Y_lst.append(pre_y)

            #  Collecting and the Truth Table of expected and predicted values.
            EY_lst.append(exp_y)

    else:
        raise Exception(
            "Invalid genre_tag argument's value: Valid arguments are either and integer \
            (as genre tag) of 'None'"
        )

    # Stacking the lists to Single arrays for PS and EY respectively
    PS = np.hstack(PS_lst)
    EY = np.hstack(EY_lst)
    PR_Y = np.hstack(PR_Y_lst)

    # Converting in to binary case under binary variable condition.
    if binary:
        EY = np.where(EY == PR_Y, 1, 0)

    # Shorting results by Predicted Scores
    inv_srd_idx = np.argsort(PS)[::-1]
    PS = PS[inv_srd_idx]
    EY = EY[inv_srd_idx]
    PR_Y = PR_Y[inv_srd_idx]

    # Returning Predicted Scores and Expected Values
    return (PS, EY, PR_Y)