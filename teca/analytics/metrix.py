"""
    This module included ROC, PRC, AUC, Interpolation, etc., functions.
    Details can be found in each function separately.

    Aurthor: Dimitrios Pritos

"""

import numpy as np


def roc_curve(trh_arr, scr_arr, full_curve=False, arr_type=np.float32):
    """Receiver Operating Characteristic (ROC) curves.

    Returns the Receiver Operating Characteristic curve given the expected classes table,
    i.e. the binary real labels of the samples and the scores/probabilities returned by the
    classifier, either as final result or in an indeterminate stage of the classification
    (prediction).

    This algorithm has been developed for exploiting the monotonicity of the curve, i.e. any
    instance that is classified positive with respect to a given threshold will be classified
    positive for all lower thresholds as well. Therefore, we can simply sort the test instances
    decreasing by f scores and move down the list, processing one instance at a time and updating
    TP and FP as we go. In this way an ROC graph can be created from a linear scan. Cited in
    "ROC Graphs: Notes and Practical Considerations for Researchers" by Tom Fawcett.

    Input arguments:

        trh_arr: The binary expected classes array, i.e. the REAL LABLES of the samples
            has been given to the Classifier.
            Valid values:  +1 for positive samples
                            0 or -1 for negative samples.
            arr_type: (optional) user-defined arrays type. default numpy.flaot32

        scr_arr: The Classifier's returning scores/probabilities array for samples
            being positive.

        full_curve: Expected values are {0,1} or {True, False}:
            If its values is 1 or True it will return a point for every input score in scr_arr.
            If its values is 0 or False it will return a point for only the unique input scores,
            i.e. numpy.unique(scr_arr) returning values.

    Output:

        tp_rate (Recall): True positive rate values array.
        fp_rate: False positive rate values array.
        unique_scores: Unique Scores from scr_arr argument. These values are the thresholds
            where new points in ROC curve where added.

    """

    # Weak checking for invalid values in real classes (values) array.
    if not (np.sum(trh_arr == 0) + np.sum(trh_arr == 1)) == trh_arr.size and\
            not (np.sum(trh_arr == -1) + np.sum(trh_arr == 1)) == trh_arr.size:
        raise Exception("Samples truth table array contains invalid values:\
                    Only {1,0} or {1,-1} (float or integer) sets of values are valid.")

    # Checking the expected values (True/False) for 'full_curve' argument
    if not isinstance(full_curve, bool) and full_curve not in [0, 1]:
        raise Exception("full_curve argument expected values are: Only {0,1} or {True,False}.")

    # In place conversion of numerical type in case the input is in integer.
    trh_arr = trh_arr.astype(arr_type, copy=False)
    scr_arr = scr_arr.astype(arr_type, copy=False)

    # Convert truth binary array from {1, -1} to {1, 0}.
    trh_arr[np.where(trh_arr == -1)] = 0

    # Counting the total number of positive and negative samples.
    pos_sum = float(trh_arr.sum())
    neg_sum = float(trh_arr.size - pos_sum)

    # Initialising True Positive and False Positive counters.
    tp = 0
    fp = 0

    # Initialising True Positive and False Positive rates.
    tp_rate = list()
    fp_rate = list()

    # Initialising last score.
    last_scr = -1

    # Building the ROC curve
    for exp_y, scr in zip(trh_arr, scr_arr):

        if exp_y > 0:  # if expected y is 1
            tp += 1
        else:  # if expected y is 0
            fp += 1

        if scr != last_scr or full_curve:
            tp_rate.append(tp / pos_sum)
            fp_rate.append(fp / neg_sum)
            last_scr = scr

    # Appending last point if not already
    tp_rate.append(tp / pos_sum)
    fp_rate.append(fp / neg_sum)

    # Converting TP-Rate and FP-Rate lists to numpy.arrays
    tp_rate = np.array(tp_rate, dtype=arr_type)
    fp_rate = np.array(fp_rate, dtype=arr_type)

    # Returning the ROC curve
    return tp_rate, fp_rate, np.unique(scr_arr)


def pr_curve(trh_arr, scr_arr, full_curve=False, is_truth_tbl=False, arr_type=np.float32):
    """Precision-Recall (PR) curves.

    Returns the Precision-Recall curve given the truth table, i.e. the binary real labels of the
    samples and the scores/probabilities returned by the classifier, either as final result or in
    an indeterminate stage of the classification/prediction.

    This algorithm has been developed for exploiting the monotonicity of the curve,
    i.e. any instance that is classified positive with respect to a given threshold will be
    classified positive for all lower thresholds as well. Therefore, we can simply sort the test
    instances decreasing by f scores and move down the list, processing one instance at a time and
    updating TP as we go. In this way an PR graph can be created from a linear scan. Following the
    same line of thought as the algorithm is described for ROC curves which has been cited in
    "ROC Graphs: Notes and Practical Considerations for Researchers" by Tom Fawcett.

    Input arguments:

        trh_arr: The binary REAL LABLES array, i.e. the real classes of the samples
            has been given to the Classifier.
            (*)Alternatively, given the flag is_truth_tbl == True the trh_arr can be the
            TRUTH TABLE of predictions, i.e. that is the results of Expected Y == Predicted Y.
            Valid values:  +1 for positive samples.
                            0 or -1 for negative samples.
            arr_type: (optional) user-defined arrays type. default numpy.flaot32

        scr_arr: The Classifier's returning scores/probabilities array for samples
            being positive.

        full_curve: Expected values are {0,1} or {True, False}:
            If its values is 1 or True it will return a point for every input score in scr_arr.
            If its values is 0 or False it will return a point for only the unique input scores,
            i.e. numpy.unique(scr_arr) returning values.

        is_truth_tbl: This flag indicates whether the positive sum will be the one given form the
            Ground Truth or it will be equal to the length of the trh_arr (when this is a Truth
            Table of precisions).

    Output:

        precision: Precision values array.
        recall: Recall values array (equivalent to the True positive rate).
        tp_rate: False positive rate values array.
        unique_scores: Unique Scores from scr_arr argument. These values are the thresholds
            where new points in PR curve where added.

    """

    # Checking for invalid values in real classes (values) array.
    if not (np.sum(trh_arr == 0) + np.sum(trh_arr == 1)) == trh_arr.size and\
            not (np.sum(trh_arr == -1) + np.sum(trh_arr == 1)) == trh_arr.size:
        raise Exception("Samples truth table array contains invalid values:\
                    Only {1,0} or {1,-1} (float or integer) sets of values are valid.")

    # Checking the expected values (True/False) for 'full_curve' argument
    if not isinstance(full_curve, bool) and full_curve not in [0, 1]:
        raise Exception("full_curve argument expected values are: Only {0,1} or {True,False}.")

    # In place conversion of numerical type in case the input is in integer.
    trh_arr = trh_arr.astype(arr_type, copy=False)
    scr_arr = scr_arr.astype(arr_type, copy=False)

    # Convert truth binary array from {1, -1} to {1, 0}.
    trh_arr[np.where(trh_arr == -1)] = 0

    # Counting the total number of positive.
    if is_truth_tbl:
        # This is actually correct only when the input is the truth table of predictions.
        pos_sum = float(trh_arr.shape[0])

    else:
        # This is the correct for binary input, i.e. the Ground Truth is used as input argument.
        pos_sum = float(trh_arr.sum())

    # Initialising True Positive and False Positive counters.
    tp = 0

    # Initialising True Positive and False Positive rates.
    precision = list()
    recall = list()

    # Appending fist fixed point (x, y) = (0, 1).
    # This point might be a duplicate in best case or misleading in worst case.
    precision.append(1.0)
    recall.append(0.0)

    # Initialising last score and document counter.
    last_scr = -1
    doc_cnt = 0.0

    # Building the PR curve
    for exp_y, scr in zip(trh_arr, scr_arr):

        doc_cnt += 1.0

        if exp_y > 0:  # if expected y is 1
            tp += 1

        if scr != last_scr or full_curve:
            precision.append(tp / doc_cnt)
            recall.append(tp / pos_sum)
            last_scr = scr

    # Appending last point if not already
    precision.append(tp / doc_cnt)
    recall.append(tp / pos_sum)

    # Converting Precision and Recall lists to numpy.arrays
    precision = np.array(precision, dtype=arr_type)
    recall = np.array(recall, dtype=arr_type)

    # Returning the ROC curve
    return precision, recall, np.unique(scr_arr)


def pr_curve_macro(exp_y, pre_y, scrz, full_curve=False, arr_type=np.float32):
    """Macro Precision-Recall (PR) curves for Multi-Class classification algorithms' output.

    Returns the Macro Precision-Recall curve of an multiclass classfication algorithm's outpout.
    This algorithm is calcilating the Contigency-table (Confusion matrix) for all the classifier's
    predicitons up to the current sample's predictions where the score for the prediciton changes.
    However, it is possible given the full_curve agument equal to True value, to get the
    Macro-Precision and the Macro-Recall for every sample irrespectively of the prediction scores.
    This algorithm has been developed for exploiting the monotonicity of the curve,
    i.e. any instance that is classified positive with respect to a given threshold will be
    classified positive for all lower thresholds as well.
    Input arguments:

        exp_y: A vector of the Multi-Class tags expected for each sample. Zero (0) is not permited
            because it is used when unknon-class case is enabled in the seq_contingency_table()
            function which is used for calculating the contigency table while the preidction
            are given sequentialy.

        pre_y: A vector of Multi-Class tags predicted for each sample by the classifier.

        scrz: A vector of scores (similarity, certainty, etc.) predicted for each sample
            by the classifier.

        full_curve: Expected values are {0,1} or {True, False}:
            If its values is 1 or True it will return a point for every input score in scrz.
            If its values is 0 or False it will return a point for only the unique input scores,
            i.e. numpy.unique(scrz) returning values.

    Output:

        precision: Macro Precision values array.
        recall: Macro Recall values array.
        tp_rate: False positive rate values array.
        unique_scores: Unique Scores from scrz argument. These values are the thresholds
            where new points in PR curve where added.

    """

    # Checking the expected values (True/False) for 'full_curve' argument
    if not isinstance(full_curve, bool) and full_curve not in [0, 1]:
        raise Exception("full_curve argument expected values are: Only {0,1} or {True,False}.")

    # In place conversion of numerical type in case the input is in integer.
    exp_y = exp_y.astype(arr_type, copy=False)
    pre_y = pre_y.astype(arr_type, copy=False)
    scrz = scrz.astype(arr_type, copy=False)

    # Initialising True Positive and False Positive rates.
    precision = list()
    recall = list()

    # Appending fist fixed point (x, y) = (0, 1).
    # This point might be a duplicate in best case or misleading in worst case.
    precision.append(1.0)
    recall.append(0.0)

    # Initialising last score and document counter.
    last_scr = -1

    # Getting the expected classes.
    exp_cls_tags_set = np.unique(exp_y)

    # Getting the predected classes.
    pre_cls_tags_set = np.unique(pre_y)

    # Getting the number of samples per class. Zero tag is inlcuded.
    smpls_per_cls = np.bincount(np.array(exp_y, dtype=np.int))

    # Keeping from 1 to end array in case the expected class tags start with above zero values.
    if smpls_per_cls[0] == 0 and exp_cls_tags_set[0] > 0:
        smpls_per_cls = smpls_per_cls[1::]
    else:
        raise Exception("Samples count in zero bin is different to the expected class tag count!")

    # Building the PR curve
    for i, scr in enumerate(scrz):

        # Getting the class tags predicted so far.
        crnt_prcls_num = np.unique(pre_y[:i+1]).shape[0]

        conf_mtrx = seq_contingency_table(
            exp_y[:i+1], pre_y[:i+1],
            exp_cls_tags_set=exp_cls_tags_set, pre_cls_tags_set=pre_cls_tags_set,
            arr_type=arr_type
        )

        if scr != last_scr or full_curve:

            # print conf_mtrx, np.sum(conf_mtrx, axis=1),
            # print exp_known_cls_num, smpls_per_cls, crnt_prcls_num, crnt_prtgs

            # Calculating Macro-Precision.
            precision.append(
                np.sum(
                    [dg / float(pred_docs)
                        for dg, pred_docs in zip(np.diag(conf_mtrx), np.sum(conf_mtrx, axis=1))
                        if pred_docs > 0]
                ) / crnt_prcls_num
            )

            # Calculating Macro-Recall.
            recall.append(
                np.sum(
                    [dg / float(splpc)
                        for dg, splpc in zip(np.diag(conf_mtrx), smpls_per_cls)
                        if splpc > 0]

                ) / exp_cls_tags_set.shape[0]
            )

    # Appending last point if not already.
    # Calculating Macro-Precision.
    precision.append(
        np.sum(
            [dg / float(pred_docs)
                for dg, pred_docs in zip(np.diag(conf_mtrx), np.sum(conf_mtrx, axis=1))
                if pred_docs > 0]
        ) / crnt_prcls_num
    )

    # Calculating Macro-Recall.
    recall.append(
        np.sum(
            [dg / float(splpc)
                for dg, splpc in zip(np.diag(conf_mtrx), smpls_per_cls)
                if splpc > 0]

        ) / exp_cls_tags_set.shape[0]
    )

    """
    print 'DIAG', np.diag(conf_mtrx)
    print 'SAMPLS', smpls_per_cls
    print zip(np.diag(conf_mtrx), smpls_per_cls)
    print 'RES', [dg / float(splpc)
        for dg, splpc in zip(np.diag(conf_mtrx), smpls_per_cls)
        if splpc > 0]
    print 'RES', [dg / float(pred_docs)
        for dg, pred_docs in zip(np.diag(conf_mtrx), np.sum(conf_mtrx, axis=1))
        if pred_docs > 0]
    print 'RES', np.mean([dg / float(splpc)
        for dg, splpc in zip(np.diag(conf_mtrx), smpls_per_cls)
        if splpc > 0])
    print 'RES', np.mean([dg / float(pred_docs)
        for dg, pred_docs in zip(np.diag(conf_mtrx), np.sum(conf_mtrx, axis=1))
        if pred_docs > 0])
    """

    # Converting Precision and Recall lists to numpy.arrays
    precision = np.array(precision, dtype=arr_type)
    recall = np.array(recall, dtype=arr_type)

    # Returning the ROC curve
    return precision, recall, np.unique(scrz)


def auc(x, y, is_zcc=True, arr_type=np.float32):
    """Area Under the Curve (AUC).

    Returns the Area Under the Curve of a given curve (ROC, PR, etc.). The trapezoid approximation
    is used as instructed in several tutorial and papers such as "ROC Graphs: Notes and Practical
    Considerations for Researchers" by Tom Fawcett.

    It performed an ascending order checking on curve's coordinates sequences assuring that
    the results will always be correct. In case the order is not in ascending order the points
    are reordered.

    Input arguments:

        x: is the numpy.array sequence of all x coordinates of the curve's points.
        y: is the numpy.array sequence of all y coordinates of the curve's points.
        is_zcc: (optional) user-defined whether the input curve is zero-padded....
        ...In addiont with this option the curve is assumed to be Continuous. Default value is True
        arr_type: (optional) user-defined arrays type. default numpy.flaot32

    Output:

        auc: a floating point value equal to the sum of trapezoids formed
            by the curve's points given as arguments.

    """

    # Checking the validity of the x and y arrays before start computing the AUC.
    if x.size != y.size:
        raise Exception("X and Y coordinate arrays must have equal length.")

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise Exception("Invalid array-argument's Dimensions: Only 1D arrays are acceptable\
                    for this implementation of computing AUC.")

    if x.size < 2:
        raise ValueError("AUC cannot computed given a single point.")

    # In place conversion of numerical type in case the input is in integer.
    x = x.astype(arr_type, copy=False)
    y = y.astype(arr_type, copy=False)

    # Checking curve's points correct sequence.
    if np.argmax(x) == 0:  # Sequence assumed to be in descending order so it is inverted.
        y = y[::-1]
        x = x[::-1]

    # Checking the Sequence. If x it is in random order the input is not a proper curve.
    if not np.array_equal(np.sort(x), x) and not np.array_equal(np.sort(x), x[::-1]):
        raise Exception(
            "X-coordinate sequence is in random order: Impossible to calculate the AUC correctly.")

    # NOTE: This is option assuming that the curve is Continuous and Zero padded
    if is_zcc:
        y = y[np.where(y != 0.0)[0]]
        x = x[np.where(y != 0.0)[0]]

    # Calculating the delta-x's, i.e. bases of trapezoids.
    dx = np.abs(np.diff(x))

    # Calculating the average hight amongst all sequential pairs of y axis heights.
    height_means = (y[1:] + y[:-1]) / 2.0

    # Returning the AUC, i.e. the sum of all trapezoids formed under the curve.
    # This is equal to matrix operation h*b.T or array operation sum(h*b).
    return np.sum(height_means * dx)


def smooth_linear(y, x=None, arr_type=np.float32):
    """Practical linear smoothing function.

    Takes as argument a saw-like shaped curve and returns the curve connecting the local maxima of
    the curve. There is no need for X axis values to be given in case only the Y axis values are
    required for all x values.

    In case the x values are provided then only the y thresholds (local maxima) with their
    respective x values will be returned.

    NOTE: ***The Y sequence should be inverted from the lowest to the highest values ***
          *** I works properly for curves that increasing or decreasing for all x values
              (Think about it by studying the code bellow)***

    Input arguments:

        y: is the numpy.array sequence of all y coordinates of the curve's points.
        x: is the numpy.array sequence of all x coordinates of the curve's points.
        arr_type: (optional) user-defined arrays type. default numpy.flaot32

    Output:

        smooth_y: is the numpy.array sequence of all local maxima of y axis.
        smooth_x: is the numpy.array sequence of the respective x of smooth_y sequence.
                  *(only when x values sequence is provided as input argument.)

    """

    # The list of local maxima. Either for each x value or only for the unique local maxima.
    smooth_y = list()

    # Finding the unique local maxima.
    # The y values sequence assumed to be given from lower to higher value
    max_y = -1
    for yval in y:

        # The same local maxima is appended to the list until a greater value is occurring.
        if yval > max_y:
            max_y = yval

        smooth_y.append(max_y)

    # Converting the list to an numpy array using the proper data type given as argument.
    smooth_y = np.array(smooth_y, dtype=arr_type)

    # Finding the thresholds and their respective x values in case x sequence is given as argument.
    if isinstance(x, np.ndarray) or isinstance(x, list):

        # Getting the unique maximum values.
        y_thresholds, u_inds = np.unique(smooth_y, return_index=True)

        # Getting the respective x for each threshold.
        smooth_x = np.array(x, dtype=arr_type)[u_inds]

        # Returning the values.
        return y_thresholds, smooth_x

    # Returning the smoothed y values sequence where the values between the local maxima
    # has been replaced with the local maxima values, whenever a new one has been occurring
    return smooth_y


def zero_padding_PRC(P, R):
    """ Zero Padding function for PRCs

        This function is getting PRC and is padding Y with Zeros and extends recall with the
        rest of values are missing. The step (resolution) is used for the padding id the same
        with the one occurs based on the original curve. However, is getting a mean distance
        and not the exact distance between P and R values that might variate.

        Input arguments:

        P: is the numpy.array sequence of all y coordinates of the curve's points.
        R: is the numpy.array sequence of all x coordinates of the curve's points.

    Output:

        P: is the padded P, an numpy.array sequence of all y coordinates of the curve's points.
        R: is the padded R, an numpy.array sequence of all x coordinates of the curve's points.

    """

    # Padding Recall and precision with proper resolution.
    # Finding the max recall level.
    max_rec_lvl = np.max(R)

    # Calculating resolution.
    pding_step = max_rec_lvl / float(len(R))

    # Some times in seems that the step is becoming 0.0 thus we replace it with 0.01.
    if not pding_step:
        pding_step = 0.01

    # Padding Recall sequence.
    R_pad = np.arange(max_rec_lvl + pding_step, 1 + pding_step, pding_step)
    R = np.hstack((R, R_pad))

    # Padding Precision sequence.
    P_pad = np.zeros_like(R_pad)
    P = np.hstack((P, P_pad))

    # Returning zero padded P and R
    return P, R


def reclev11_averaging(P, R, rcl_tuple=(0.0, 1.1, 0.1)):
    """Recall Level Averaging function.

    It takes the Precision Recall (RP) Curve and returns an averaged PR curve at particular points
    as given by the user. The same applies for any curve for which one would like to average it at
    any particular X positions.

    Input arguments:

        P: is the numpy.array sequence of all y coordinates of the curve's points.
        R: is the numpy.array sequence of all x coordinates of the curve's points.
        rcl_tuple: A three elements tuple for x position to be averaged where fist is the
            starting position, the second is the ending position and the third is the
            increment step.

    Output:

        avg_P: is the numpy.array Precision values sequence of the Averaged PR curve.
        R_Levels: is the numpy.array of the Recall levels created based rcl_tuple argument
            or default, i.e. the 11 Recall Levels.

    """

    # Creating the array of recall levels for which the curve will be averaged.
    # Default: 11 recall levels (0 to 10).
    if len(rcl_tuple) != 3:
        raise Exception("Two (2) or three (3) arguments are expected an input for this function.")
    R_Levels = np.arange(rcl_tuple[0], rcl_tuple[1], rcl_tuple[2])

    # Padding Recall and precision with proper resolution.
    P, R = zero_padding_PRC(P, R)

    # The Array for storing the Precision values respectively to the recall levels.
    avg_P = np.zeros_like(R_Levels, dtype=np.float)

    # Init the smallest index from where the part of the line, which will be
    # used to be averaged, starts.
    last_rl_idx = 0

    # The initial value for this variable should equal to be 1 because in PR
    # curves the (0,1) point is fixed.
    last_avg_P = P[0]

    # Averaging the part of the PR Curve at each level of the given recall
    # levels (default 11 levels).
    for i, r in enumerate(R_Levels[0:-1]):

        # Getting the closest index to the recall level of the current loop.
        # In particular the largest index is selected in case we have more than
        # one position with the same small distance.
        current_rl_idx = np.max(np.where(np.abs(R - r) == np.min(np.abs(R - r))))

        # Alternatively it could be used the first occurred index respective to the minimum
        # distance. However, the above approach returns an averaged curve close to the real one
        # (with full points).
        # current_rl_idx = (np.abs(R - r)).argmin()

        # Averaging each part of the line from the last recall level to the current one.
        avg_P[i] = np.mean(P[last_rl_idx: current_rl_idx])

        # Preventing the case that averaging returns NaN when 'last' and 'current'
        # indices coincide.
        if np.isnan(avg_P[i]):
            avg_P[i] = last_avg_P
        else:
            last_avg_P = avg_P[i]

        # Making the current highest index of the current loop to be the smallest one.
        last_rl_idx = current_rl_idx

    # Averaging the last part of the line. Letting the last point to be calculated
    # outside of the above loop it is assured that all point of the last part of the curve are
    # taken in to account.
    avg_P[-1] = np.mean(P[last_rl_idx::])

    return avg_P, R_Levels


def reclev11_nearest(P, R, rcl_tuple=(0.0, 1.1, 0.1)):
    """Nearest to the Recall Levels function.

    It takes the Precision Recall (RP) Curve and returns the Y values of the PR curve of the
    nearest (x,y) points to the recall levels (i.e. X values) as given by the user. The same
    applies for any curve for which one would like to average it at any particular X positions.

    Input arguments:

        P: is the numpy.array sequence of all y coordinates of the curve's points.
        R: is the numpy.array sequence of all x coordinates of the curve's points.
        rcl_tuple: A three elements tuple for x position to be averaged where fist is the
            starting position, the second is the ending position and the third is the
            increment step.

    Output:

        avg_P: is the numpy.array Precision values of the nearest to the R levels.
        R_Levels: is the numpy.array of the Recall levels created based rcl_tuple argument
            or default, i.e. the 11 Recall Levels.

    """

    # Creating the array of recall levels for which the curve will be averaged.
    # Default: 11 recall levels (0 to 10).
    if len(rcl_tuple) != 3:
        raise Exception("Two (2) or five (5) arguments are expected an input for this function.")
    R_Levels = np.arange(rcl_tuple[0], rcl_tuple[1], rcl_tuple[2])

    # Padding Recall and precision with proper resolution.
    P, R = zero_padding_PRC(P, R)

    # Init with zeros the array for saving the nearest recall levels indexes.
    idxs = np.zeros_like(R_Levels, dtype=np.int)

    for i, r in enumerate(R_Levels):
        # Getting the closest index to the recall level of the current loop.
        idxs[i] = (np.abs(R - r)).argmin()

    # Using the above indexes for selecting the Y (i.e. precision) values from
    # the PR curve to be returned.
    return P[idxs], R_Levels


def reclev11_max(P, R, rcl_tuple=(0.0, 1.1, 0.1), trec=True):
    """Maximum Y above each Recall Levels in the sequence, function.

    # NOTE: This is the proper way to find the recall levels using the one of the two formulas
    based on the instruction given by TREC or other IR conferences and journals.
        TREC: max(p(r)), r >= (rj)
        Others: max(p(r)), (rj) <= r <= (rj+1)

    It takes the Precision Recall (RP) Curve and returns the maximum Y values found for
    recall levels of the current recall level and above in the sequence. TREC's formula is
    used.

    Input arguments:

        P: is the numpy.array sequence of all y coordinates of the curve's points.
        R: is the numpy.array sequence of all x coordinates of the curve's points.
        rcl_tuple: a three elements tuple for x position to be averaged where fist is the
            starting position, the second is the ending position and the third is the
            increment step.
        trec: is the Boolean flag whether the TREC algorithm is used or not.

    Output:

        avg_P: is the numpy.array Maximum Precision values between R levels.
        R_Levels: is the numpy.array of the Recall levels created based rcl_tuple argument
            or default, i.e. the 11 Recall Levels.

    """

    # Creating the array of recall levels for which the curve will be averaged.
    # Default: 11 recall levels (0 to 10).
    if len(rcl_tuple) != 3:
        raise Exception("Two (2) or five (5) arguments are expected an input for this function.")
    R_Levels = np.arange(rcl_tuple[0], rcl_tuple[1], rcl_tuple[2])

    # Padding Recall and precision with proper resolution.
    P, R = zero_padding_PRC(P, R)

    # Fixing some numerical inconsistencies. DON'T remove it!
    R = np.array(R, dtype=np.float32)
    P = np.array(P, dtype=np.float32)

    # Init with zeros the array for saving the highest Precision for every R level.
    max_P = np.zeros_like(R_Levels, dtype=np.float)

    # Defining interpolation condition. TREC based or not.
    if trec:
        cond = lambda r, R: np.where(R >= r)
    else:
        cond = lambda r, R: np.where((R >= r) & (R <= r + 0.1))

    # Getting the maximum value of P from all the recall levels larger than current.
    # Sequence starts form 0.1 and stops at 1. Or it follows the rcl_tuple argument instructions.
    for i, r in enumerate(R_Levels):

        # Getting all indices above r and bellow (r + 0.1), when TREC condition is not used.
        idx_abv = cond(r, R)

        # Saving the maximum precisions
        max_P[i] = np.max(P[idx_abv])

    # Returning maximum Precisions fore 11 recall levels.
    return max_P, R_Levels


def contingency_table(expd_y, pred_y, unknown_class=False, arr_type=np.float32):
    """Contingency table building the function.

    It takes the expected Y and the predicted Y values of a Classifier and it is returning the
    contingency_table. The number of classes are detected by the different tags (i.e. integers)
    located in the expected Y list/array argument. In case tags others than the ones in the
    expected Y list are found into the predictions list then a Zero Class (i.e. Don't Know) will
    considered fist-in-order of rows and columns respectively.

    Input arguments:

        expd_y: is the numpy.array or python list contains the expected prediction, i.e. the real
            classes.

        pred_y: is the numpy.array or python list contains the instance prediction of the
            classifier.

        unknown_class: (optional) defines whether class tags not occuring in the Expected class tags
            vector will be considdred as unknonw-class counts or treated as an invalid value of
            a closed-set algorithm's prediction. In this case all the class tags are shifted
            properly for letting the 0 row and 0 column to be the unknonw-class tag
            counts position(s).
            *Defauls value: False

        arr_type: (optional) user-defined arrays type. default numpy.flaot32

    Output:

        conf_matrix: is the Confusion Matrix / Contingency Table returned by this function.

    """

    expd_y = np.array(expd_y, dtype=arr_type)
    pred_y = np.array(pred_y, dtype=arr_type)

    if expd_y.shape[0] != pred_y.shape[0]:
        raise Exception(
            "Input arguments length inconsistency: expd_y and pred_y must have the same length."
        )

    if not unknown_class and not (set(pred_y) & set(expd_y) == set(pred_y)):
        raise Exception(
             "Predicted Y(s) tags list contains values not valid in expected Y(s) list tags."
        )

    # If expected Y does not containing '0' as class tag and unknonw-class prediction...
    # ...are expected then one more line and row will be appeded by swiftig the class tags by 1...
    # ...and leting 0 represent the uknown_class.
    uncl = 0
    if unknown_class and np.min(expd_y):
        uncl = 1
        # unknow_class_tags_idxs = np.where((pred_y == expd_y) == False)
        # print pred_y[unknow_class_tags_idxs] == 0

    expd_cls_tags = np.unique(expd_y)

    # Redefing the class tagging in order this function to be able to accept any numerical...
    # ...tagging irrespectively of the matrix columns and raws order.
    new_expd_cls = list()
    for i, exp_tg in enumerate(expd_cls_tags):

        expd_y[expd_y == exp_tg] = i + uncl
        pred_y[pred_y == exp_tg] = i + uncl
        new_expd_cls.append(i + uncl)

    expd_cls_tags = new_expd_cls

    conf_dim = np.max(expd_cls_tags) + 1

    conf_matrix = np.zeros((conf_dim, conf_dim), dtype=arr_type)

    for i, j in zip(pred_y, expd_y):

        if i not in expd_cls_tags and unknown_class:
            # Counting the Unknown Class predicions.
            conf_matrix[0, j] += 1

        else:
            # Counting the Known (from Expected Y), either true or false, predicions.
            conf_matrix[i, j] += 1

    return conf_matrix


def seq_contingency_table(expd_y, pred_y, exp_cls_tags_set, pre_cls_tags_set, arr_type=np.float32):
    """Sequential Contingency table building the function.

    NOTE:
        The sequential tagging should follow a specific order from 0 to X integer. In case the
        class tags are random then the function will return a false Contingency Table.

    Input arguments:

        expd_y: is the numpy.array or python list contains the expected prediction, i.e. the real
            classes.

        pred_y: is the numpy.array or python list contains the instance prediction of the
            classifier.

        exp_cls_tags: ???

        pre_cls_tags: ???

        arr_type: (optional) user-defined arrays type. default numpy.flaot32

    Output:

        conf_matrix: is the Confusion Matrix / Contingency Table returned by this function.

    """

    expd_y = np.array(expd_y, dtype=arr_type)
    pred_y = np.array(pred_y, dtype=arr_type)

    if expd_y.shape[0] != pred_y.shape[0]:
        raise Exception(
            "Input arguments length inconsistency: expd_y and pred_y must have the same length."
        )

    if pre_cls_tags_set[0] > 0 and exp_cls_tags_set[0] == 0:
        raise Exception(
            "Expected class tags set is starting with Zero(0) and Predicted class tags set is not."
        )

    # Selecting the proper size for the confusion matrix.
    if pre_cls_tags_set[0] == 0:
        conf_dim = len(pre_cls_tags_set)
    elif pre_cls_tags_set[0] > 0:
        conf_dim = len(pre_cls_tags_set) + 1
    else:
        raise Exception(
            "Invalid Predicted class tags set."
        )

    # Initializing the confusion matrix.
    conf_matrix = np.zeros((conf_dim, conf_dim), dtype=arr_type)

    # Filling in the confusion matrix.
    for i, j in zip(pred_y, expd_y):
        # Counting true and false predicions.
        conf_matrix[i, j] += 1

    # Keeping only the confusion matrix part where only the Real/Expected class tags are included...
    # ...i.e. excluding the Unknon-class tags zero(0) when it is not includetd in the expected_y...
    # ...vector.
    if pre_cls_tags_set[0] == 0 and exp_cls_tags_set[0] > 0:
        conf_matrix = conf_matrix[1::, 1::]

    return conf_matrix


def precision_recall_scores(conting_tbl, arr_type=np.float32):
    """Precision and Recall scores. (it requires a contingency table as an input)

    It takes a contingency table (i.e. a confusion matrix) as an input and returns the precision
    and recall scores for each class in 2D array. In the precision recall table row index are
    respective to the classes assigned to the contingency tables.

    Input arguments:

        contig_tbl: is the contingency table (i.e. confusion matrix).
        arr_type: (optional) user-defined arrays type. default numpy.flaot32

    Output:

        prec_recl_scores: a table of precision and recall scores for each class of the contingency
        table.

    """

    conting_tbl = np.array(conting_tbl, dtype=arr_type)

    if conting_tbl.shape[0] != conting_tbl.shape[1]:
        raise Exception("Contingency table must be a 2D square matrix.")

    pred_cls_sums_lst = np.sum(conting_tbl, axis=1)
    expd_cls_sums_lst = np.sum(conting_tbl, axis=0)

    cls_tp_lst = np.diagonal(conting_tbl)

    prec_scores = np.array([
        cls_tp / float(pred_cls_spls)
        for cls_tp, pred_cls_spls in zip(cls_tp_lst, pred_cls_sums_lst)
        if pred_cls_spls > 0
    ])

    recl_scores = np.array([
        cls_tp / float(expd_cls_spls)
        for cls_tp, expd_cls_spls in zip(cls_tp_lst, expd_cls_sums_lst)
        if expd_cls_spls > 0
    ])

    return prec_scores, recl_scores


def bcubed_pr_scores(clstrs_y, cats_y, arr_type=np.float32):
    """BCubed Precision and Recall Scores.

    The BCubed Precision and Recall scores are measures for estimating the performance of a
    clustering task. In particular BCubed-Precision of an item is the proportion of items in its
    cluster which have the item's category (including itself). The BCubed-Recall of items in its
    category which have the item's clusters (including itself). The overall BCubed Precision and
    the overall BCubed recall is the averaged respective score of all items in the distribution.
    Since the average is calculated over items, it is not necessary to apply any weighting
    according to the size of clusters or categories. These scores calculation is implemented based
    on the formal definition of them cited in Amigo et al. 2009.

    Arguments
    ---------
        clstrs_y: An array of all the clusters tags assigned to each data point by the clustering
                  algorithm and the indeces of the array are indicating the respective data points.
        cats_y: An array of all the category tags expected for each data point by assigned by the
                clustering algorithm. The data points indeces expected to be with the same order as
                the clstrs_y array.

    Output
    ------
        pre_bc: The overall BCubed-Precision.
        rec_bc: The overall BCubed-Recall.

    """

    if clstrs_y.shape[0] != cats_y.shape[0]:
        raise Exception('Clusters tags and expected tags vectors should have the same size of' +
                        ' elements, however, not necessarily the same tags (or same kind).')

    size_per_clstr = np.bincount(clstrs_y)
    size_per_cats = np.bincount(cats_y)
    # print size_per_clstr, size_per_cats

    ith_pre_bc = np.zeros_like(clstrs_y, dtype=arr_type)
    ith_rec_bc = np.zeros_like(clstrs_y, dtype=arr_type)

    for i, (clr_y, cat_y) in enumerate(zip(clstrs_y, cats_y)):

        # Finding the number of elements having the same cluster tag as the i'th data point.
        ith_same_clstr_cntr = np.where(clstrs_y == clr_y)[0]

        # Finding the number of elements having the same category tag as the i'th data point.
        ith_same_cat_cntr = np.where(cats_y == cat_y)[0]

        # Counting the data points correctly clustered, i.e. the amount of array indeces...
        # ...returned for the cluster tags being the same as the ones returned for the...
        # ...category tags.
        truth_arr_clr_cat = np.bincount(np.in1d(ith_same_clstr_cntr, ith_same_cat_cntr))[1]

        # Calculating the BCubed Precision and Recall scores for the i'th data point.
        ith_pre_bc[i] = truth_arr_clr_cat / float(size_per_clstr[clr_y])
        ith_rec_bc[i] = truth_arr_clr_cat / float(size_per_cats[cat_y])

    # Calculating the Overall BCubed Precision and recall.
    pre_bc = np.mean(ith_pre_bc)
    rec_bc = np.mean(ith_rec_bc)

    # Returning Overall Bcubed Precision and Overall Bcubed Recall.
    return pre_bc, rec_bc, size_per_clstr, size_per_cats


def bcubed_pr_curves(clsrts_arr, scrs_arr,
                     full_curve=False, is_truth_tbl=False, arr_type=np.float32):
    """
    """
    pass


def openness(tn, ts, tg):
    """ Returing the Openess Score.

    Openness score based on the definision found in Scheirer2012 'Towards Open Set Recognition'

    Arguments
    ---------
        tn: Training set size, i.e. number of classes/clusters known given for training.
        ts: Test set size, i.e. number of classes/clusters known and unkown given for test.
        tg: Target set szie, i.e. number of classes/clusters known.

    Output
    ------
        The Openness Score.

    """

    tn = float(tn)
    ts = float(ts)
    tg = float(tg)

    return 1 - np.sqrt((2*tn)/(ts+tg))


# Pure Python Implementation
class purepy(object):

    @staticmethod
    def roc_curve(truth_d, scr_d):
        """Receiver Operating Characteristic (ROC)"""

        pos_sum = sum([y == 'Y' for y in truth_d.values()])
        neg_sum = len(truth_d) - pos_sum

        pos_cnt = 0
        neg_cnt = 0

        scr_ybin_lst = list()
        for key, bin_val in truth_d.items():

            if key in scr_d:
                bin_int = 1 if bin_val == 'Y' else 0
                scr_ybin_lst.append((scr_d[key], bin_int))
            else:
                # add no provided answers with negative value in Ground-Truth file
                neg_cnt += 1 if bin_val == 'N' else 0

        scr_ybin_srd_lst = sorted(
            scr_ybin_lst, key=lambda scr_ybin_lst: scr_ybin_lst[0], reverse=True)

        tp_rate = list()
        fp_rate = list()

        last_scr = -1
        # append [0, neg_cnt / float(neg_sum)]
        # tp_rate.append( 0.0 )
        # fp_rate.append( neg_cnt / float(neg_sum) )

        for i, (scr, y) in enumerate(scr_ybin_srd_lst):

            if scr != last_scr:
                tp_rate.append(pos_cnt / float(pos_sum))
                fp_rate.append(neg_cnt / float(neg_sum))
                last_scr = scr

            if int(y) == 1:
                pos_cnt += 1
            elif int(y) == 0 or int(y) == -1:
                neg_cnt += 1
            else:
                raise Exception("Incompatible input: -1, 0, 1 values supported for Y binary")

        # Append last point if not already
        tp_rate.append(pos_cnt / float(pos_sum))
        fp_rate.append(neg_cnt / float(neg_sum))

        norm = 1.0  # float(pos_sum*neg_sum)

        return tp_rate, fp_rate, norm

    @staticmethod
    def auc(roc_curve):

        if isinstance(roc_curve, str):
            roc_curve = list(eval(roc_curve))

        y = roc_curve[0]
        x = roc_curve[1]

        dx = [float(x1 - x2) for x1, x2 in zip(x[1::], x)]
        dx0 = [0]
        dx0.extend(dx)

        h = [(y1 + y2) / 2.0 for y1, y2 in zip(y[1::], y)]
        h0 = [0]
        h0.extend(h)

        return sum([dx * y for dx, y in zip(dx0, h0)]) / roc_curve[2]

if __name__ == '__main__':

    expd_y = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    pred_y = np.array([0, 1, 2, 1, 2, 1, 2, 3, 1, 2])
    expd_cls_num = 3

    print seq_contingency_table(expd_y, pred_y, expd_cls_num, unknown_class=True)
