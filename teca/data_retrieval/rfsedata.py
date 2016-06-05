"""
    This module includes RFSE data retrieval functions.
    Details can be found in each funciton separatly.

    Aurthor: Dimitrios Pritos

"""

import numpy as np
import sys
sys.path.append('../../src')


def multiclass_res(res_h5file, kfolds, params_path, binary=None, strata=None):
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
        PY:

    """

    # Initializing.
    PS_lst = list()  # Ensemble Predicted Scores
    EY_lst = list()  # Ensemble Truth Table
    PY_lst = list()  # Ensemble Predicted Y's

    # Collecting Scores for and Expected Values for every fold given in kfold list.
    for k in kfolds:

        # Loading predicted scores.
        pre_score = res_h5file.get_node(
            params_path + '/KFold' + str(k), name='predicted_scores'
        ).read()

        # Getting the Expected genre tags.
        exp_y = res_h5file.get_node(params_path + '/KFold' + str(k), name='expected_Y').read()

        # Getting the Expected genre tags.
        pre_y = res_h5file.get_node(params_path + '/KFold' + str(k), name='predicted_Y').read()

        # Making a stratified selection upon a specific group of the results.
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
        PY_lst.append(pre_y)

        #  Collecting and the Truth Table of expected and predicted values.
        EY_lst.append(exp_y)

    # Stacking the lists to Single arrays for PS and EY respectively
    PS = np.hstack(PS_lst)
    EY = np.hstack(EY_lst)
    PY = np.hstack(PY_lst)

    # Converting into binary case under binary variable condition.
    if binary:
        EY = np.where(EY == PY, 1, 0)

    # Shorting results by Predicted Scores
    inv_srd_idx = np.argsort(PS)[::-1]
    PS = PS[inv_srd_idx]
    EY = EY[inv_srd_idx]
    PY = PY[inv_srd_idx]

    # Returning Predicted Scores and Expected Values and Predicted values
    return (PS, EY, PY)


def onevsall_res(res_h5file, genre_tag, kfolds, params_path):

    # Initializing.
    PS_lst = list()  # Ensemble Predicted Scores
    EY_lst = list()  # Ensemble Truth Table
    PY_lst = list()  # Ensemble Predicted Y's

    # Getting simga threshold.
    sigma = float('.' + params_path.split('/')[3].split('a')[1][1::])

    # Collecting Scores for and Expected Values for every fold given in kfold list.
    for k in kfolds:

        # Getting predicited classed per iteration.
        pc_per_iter = res_h5file.get_node(
            params_path + '/KFold'+str(k), name='predicted_classes_per_iter'
        ).read()

        # Since it is a Binary case get the positive and negavtive scores. Then keeping...
        # ...the postivie scores as the Predition scores.
        gnr_pred_pos = np.where(pc_per_iter == genre_tag, 1, 0)
        pos_ps = np.sum(gnr_pred_pos, axis=0) / np.float(pc_per_iter.shape[0])

        gnr_pred_neg = np.where(pc_per_iter == genre_tag, 0, 1)
        neg_ps = np.sum(gnr_pred_neg, axis=0) / np.float(pc_per_iter.shape[0])

        # Caclulating the predicions in 1-vs-All case. Calculating the Predicted Class...
        # ...and the Sigma scores for each document.
        pre_y = np.zeros_like(pos_ps)
        pre_score = np.zeros_like(pos_ps)

        for i, (ps, ns) in enumerate(zip(pos_ps, neg_ps)):

            if ps > ns and ps >= sigma:

                pre_y[i] = 1.0
                pre_score[i] = ps

            elif ns > ps and ns > sigma:

                pre_score[i] = ns

            # else: let everythong to be Zero OR I might need to recosider it!

        #  Predicted Y(s) and Prediction Scores has been calculated on the given Genre...
        #  ...i.e. assuming that the rest genres being Negative examples,...
        #  ...based on 'sigma' threshold.
        PY_lst.append(pre_y)

        PS_lst.append(pre_score)

        # Collecting the excepted tag values by conventing them first in binary form.
        exp_y = res_h5file.get_node(params_path + '/KFold' + str(k), name='expected_Y').read()
        EY_lst.append(np.where(exp_y == genre_tag, 1, 0))

    # Stacking the lists to Single arrays for PS and EY respectively
    PS = np.hstack(PS_lst)
    EY = np.hstack(EY_lst)
    PY = np.hstack(PY_lst)

    # Shorting results by Predicted Scores
    inv_srd_idx = np.argsort(PS)[::-1]
    PS = PS[inv_srd_idx]
    EY = EY[inv_srd_idx]
    PY = PY[inv_srd_idx]

    # Returning Predicted Scores and Expected Values
    return (PS, EY, PY)


def multiclass_multimeasure_res(hf5_fl1, hf5_fl2, kfolds, params_path, binary=None, strata=None):

    # Initialising.
    PS_lst = list()
    EY_lst = list()
    PY_lst = list()

    # Getting simga threshold.
    sigma = float('.' + params_path.split('/')[3].split('a')[1][1::])

    # Collecting Scores for and Expected Values for every fold given in kfold list.
    for k in kfolds:

        # Getting the Expected genre tags
        exp_y = hf5_fl1.get_node(params_path+'/KFold'+str(k), name='expected_Y').read()

        # Getting the number of classes.
        cls_num = float(len(np.unique(exp_y)))

        # Calulating Prediction Scores for the given Gerne i.e. assuming that the rest genres
        # being Negative examples

        # File # 1
        pc_array_fl1 = hf5_fl1.get_node(
            params_path + '/KFold'+str(k), name='predicted_classes_per_iter'
        ).read()

        ms_array_fl1 = hf5_fl1.get_node(
            params_path+'/KFold'+str(k), name='max_sim_scores_per_iter'
        ).read()

        # File # 2
        pc_array_fl2 = hf5_fl2.get_node(
            params_path + '/KFold'+str(k), name='predicted_classes_per_iter'
        ).read()

        ms_array_fl2 = hf5_fl2.get_node(
            params_path+'/KFold'+str(k), name='max_sim_scores_per_iter'
        ).read()

        # Normalising max-similarity scores for being comparable irrespectively of the
        # measure has been used.
        max_ms = np.max(ms_array_fl1, axis=0)
        min_ms = np.min(ms_array_fl1, axis=0)

        if (max_ms - min_ms).all() != 0:
            # EXPLAIN HERE...
            ms_array_fl1 = (ms_array_fl1 - min_ms) / (max_ms - min_ms)

        else:
            # EXPLAIN HERE...
            mm = (max_ms - min_ms)
            mm[np.where((mm == 0))] = 0.00000000000001
            ms_array_fl1 = (ms_array_fl1 - min_ms) / mm

        max_ms = np.max(ms_array_fl2, axis=0)
        min_ms = np.min(ms_array_fl2, axis=0)

        if (max_ms - min_ms).all() != 0:
            # EXPLAIN HERE...
            ms_array_fl2 = (ms_array_fl2 - min_ms) / (max_ms - min_ms)

        else:
            # EXPLAIN HERE...
            mm = (max_ms - min_ms)
            mm[np.where((mm == 0))] = 0.00000000000001
            ms_array_fl2 = (ms_array_fl2 - min_ms) / mm

        docs_num = ms_array_fl1.shape[1]
        itrs = np.arange(ms_array_fl1.shape[0])
        ms_arr1_cls = np.hsplit(ms_array_fl1, docs_num)
        ms_arr2_cls = np.hsplit(ms_array_fl2, docs_num)
        pc_arr1_cls = np.hsplit(pc_array_fl1, docs_num)
        pc_arr2_cls = np.hsplit(pc_array_fl2, docs_num)

        # print pc_arr2_cls

        new_pc_col_lst = list()

        for i, (ms_c1, ms_c2, pc_c1, pc_c2) in enumerate(zip(
                ms_arr1_cls, ms_arr2_cls, pc_arr1_cls, pc_arr2_cls)):

            i_max_ms_col = np.argmax(np.hstack((ms_c1, ms_c2)), axis=1)

            # EXPLAIN THIS IN DETAIL...
            pc4max_ms = np.hstack((pc_c1, pc_c2))[itrs, i_max_ms_col]

            new_pc_col_lst.append(pc4max_ms)

        # Calculating the Predicted Class and the Sigma scores for each document.
        pre_y = np.zeros(docs_num)
        pre_score = np.zeros(docs_num)

        for i, pc_col in enumerate(new_pc_col_lst):

            # NOTE: Check out the 'minlength' param value.
            max_gnr_scr = np.bincount(pc_col.astype(np.int), minlength=cls_num+1) / float(len(itrs))

            if sigma < np.max(max_gnr_scr):
                pre_y[i] = np.argmax(max_gnr_scr)
                pre_score[i] = np.max(max_gnr_scr)

        # Making a statified selection upon a specific group of the results.
        if strata:

            gpr_idx = -strata[1]

            pre_score_grp1 = pre_score[0:gpr_idx]
            pre_score_grp2 = pre_score[gpr_idx::]

            pre_y_grp1 = pre_y[0:gpr_idx]
            pre_y_grp2 = pre_y[gpr_idx::]

            exp_y_grp1 = exp_y[0:gpr_idx]
            exp_y_grp2 = exp_y[gpr_idx::]

            # Perfroming Stratified selection.
            unq_pred_tgs = np.unique(pre_y_grp2)

            strata_idxs = np.hstack(
                [idx_arr[0:int(np.rint(idx_arr.shape[0]/strata[0]))] for idx_arr
                    in [np.where((pre_y_grp2 == tg))[0] for tg in unq_pred_tgs]]
            )

            pre_score = np.hstack((pre_score_grp1, pre_score_grp2[strata_idxs]))
            pre_y = np.hstack((pre_y_grp1, pre_y_grp2[strata_idxs]))
            exp_y = np.hstack((exp_y_grp1, exp_y_grp2[strata_idxs]))

        # Saving the Ensemble Prediction Scores.
        PS_lst.append(pre_score)

        # Saving the Class predictions.
        PY_lst.append(pre_y)

        # Collecting and the Truth Table of expected and predicted values.
        EY_lst.append(exp_y)

    # Stacking the lists to Single arrays for PS and EY respectively
    PS = np.hstack(PS_lst)
    EY = np.hstack(EY_lst)
    PY = np.hstack(PY_lst)

    # Converting in to binary case under binary variable condition.
    if binary:
        EY = np.where(EY == PY, 1, 0)

    # Shorting Results by Predicted Scores
    inv_srd_idx = np.argsort(PS)[::-1]
    PS = PS[inv_srd_idx]
    EY = EY[inv_srd_idx]
    PY = PY[inv_srd_idx]

    # Retunring Predicted Scores and Expected Values
    return (PS, EY, PY)


def onevsall_multimeasure_res(hf5_fl1, hf5_fl2, genre_tag, kfolds, params_path):

    # Initialising.
    PS_lst = list()
    EY_lst = list()
    PY_lst = list()

    # Getting simga threshold.
    sigma = float('.' + params_path.split('/')[3].split('a')[1][1::])

    # Collecting Scores for and Expected Values for every fold given in kfold list.
    for k in kfolds:

        # Calulating Prediction Scores for the given Gerne i.e. assuming that the rest genres
        # being Negative examples

        # File # 1
        pc_array_fl1 = hf5_fl1.get_node(
            params_path + '/KFold'+str(k), name='predicted_classes_per_iter'
        ).read()

        ms_array_fl1 = hf5_fl1.get_node(
            params_path+'/KFold'+str(k), name='max_sim_scores_per_iter'
        ).read()

        # File # 2
        pc_array_fl2 = hf5_fl2.get_node(
            params_path + '/KFold'+str(k), name='predicted_classes_per_iter'
        ).read()

        ms_array_fl2 = hf5_fl2.get_node(
            params_path+'/KFold'+str(k), name='max_sim_scores_per_iter'
        ).read()

        # Normalising max-similarity scores for being comparable irrespectively of the
        # measure has been used.
        max_ms = np.max(ms_array_fl1, axis=0)
        min_ms = np.min(ms_array_fl1, axis=0)

        if (max_ms - min_ms).all() != 0:
            # EXPLAIN HERE...
            ms_array_fl1 = (ms_array_fl1 - min_ms) / (max_ms - min_ms)

        else:
            # EXPLAIN HERE...
            mm = (max_ms - min_ms)
            mm[np.where((mm == 0))] = 0.00000000000001
            ms_array_fl1 = (ms_array_fl1 - min_ms) / mm

        max_ms = np.max(ms_array_fl2, axis=0)
        min_ms = np.min(ms_array_fl2, axis=0)

        if (max_ms - min_ms).all() != 0:
            # EXPLAIN HERE...
            ms_array_fl2 = (ms_array_fl2 - min_ms) / (max_ms - min_ms)

        else:
            # EXPLAIN HERE...
            mm = (max_ms - min_ms)
            mm[np.where((mm == 0))] = 0.00000000000001
            ms_array_fl2 = (ms_array_fl2 - min_ms) / mm

        docs_num = ms_array_fl1.shape[1]
        itrs = np.arange(ms_array_fl1.shape[0])
        ms_arr1_cls = np.hsplit(ms_array_fl1, docs_num)
        ms_arr2_cls = np.hsplit(ms_array_fl2, docs_num)
        pc_arr1_cls = np.hsplit(pc_array_fl1, docs_num)
        pc_arr2_cls = np.hsplit(pc_array_fl2, docs_num)

        # print pc_arr2_cls

        new_pc_col_lst = list()

        for i, (ms_c1, ms_c2, pc_c1, pc_c2) in enumerate(zip(
                ms_arr1_cls, ms_arr2_cls, pc_arr1_cls, pc_arr2_cls)):

            i_max_ms_col = np.argmax(np.hstack((ms_c1, ms_c2)), axis=1)

            # EXPLAIN THIS IN DETAIL...
            pc4max_ms = np.hstack((pc_c1, pc_c2))[itrs, i_max_ms_col]

            new_pc_col_lst.append(pc4max_ms)

        pc_per_iter = np.vstack(new_pc_col_lst)

        # Since it is a Binary case get the positive and negavtive scores. Then keeping...
        # ...the postivie scores as the Predition scores.
        gnr_pred_pos = np.where(pc_per_iter == genre_tag, 1, 0)
        pos_ps = np.sum(gnr_pred_pos, axis=1) / float(len(itrs))

        gnr_pred_neg = np.where(pc_per_iter == genre_tag, 0, 1)
        neg_ps = np.sum(gnr_pred_neg, axis=1) / float(len(itrs))

        # Caclulating the predicions in 1-vs-All case. Calculating the Predicted Class...
        # ...and the Sigma scores for each document.
        pre_y = np.zeros(docs_num)
        pre_score = np.zeros(docs_num)

        for i, (ps, ns) in enumerate(zip(pos_ps, neg_ps)):

            if ps > ns and ps >= sigma:

                pre_y[i] = 1.0
                pre_score[i] = ps

            elif ns > ps and ns > sigma:

                pre_score[i] = ns

            # else: let everythong to be Zero OR I might need to recosider it!

        # Saving the Ensemble Prediction Scores.
        PS_lst.append(pre_score)

        # Saving the Class predictions.
        PY_lst.append(pre_y)

        # Getting the Expected genre tags.
        exp_y = hf5_fl1.get_node(params_path+'/KFold'+str(k), name='expected_Y').read()
        EY_lst.append(np.where(exp_y == genre_tag, 1, 0))

    # Stacking the lists to Single arrays for PS and EY respectively
    PS = np.hstack(PS_lst)
    EY = np.hstack(EY_lst)
    PY = np.hstack(PY_lst)

    # Shorting Results by Predicted Scores
    inv_srd_idx = np.argsort(PS)[::-1]
    PS = PS[inv_srd_idx]
    EY = EY[inv_srd_idx]
    PY = PY[inv_srd_idx]

    # Retunring Predicted Scores and Expected Values
    return (PS, EY, PY)
