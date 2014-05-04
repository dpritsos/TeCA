""" 
    This module included ROC, PRC, AUC, Interpolation, etc., functions.
    Details can be found in each funciton separatly. 

    Aurthor: Dimitrios Pritos

"""

import numpy as np


def roc_curve(trh_arr, scr_arr, full_curve=False, arr_type=np.float32):
    """Receiver Operating Characteristic (ROC) curves.

    Returns the Receiver Operating Characteristic curve given the truth table, 
    i.e. the binary real labels of the samples and the scores/probabilities returned 
    by the classifier, either as final result or in an indeterminate stage of the 
    classification/prediction.

    This algorithm has been developed for exploiting the monotonicity of the curve, 
    i.e. any instance that is classified positive with respect to a given threshold 
    will be classified positive for all lower thresholds as well. Therefore, we can 
    simply sort the test instances decreasing by f scores and move down the list, 
    processing one instance at a time and updating T P and F P as we go. In this way
    an ROC graph can be created from a linear scan. Cited in "ROC Graphs: Notes 
    and Practical Considerations for Researchers" by Tom Fawcett.
    
    Input argumens:
        
        trh_arr: The binay truth array, i.e. the real classies of the samples 
            has been given to the Classifier.
            Valid values:  +1 for positive samples.
                            0 or -1 for negative samples.
            arr_type: (optional) user-defined arrays type. default numpy.flaot32

        scr_arr: The Classifier's returning scores/probabilities array for samples 
            being positive.

        full_curve: Expected values are {0,1} or {True, False}:
            If its values is 1 or True it will return a point for every input score in scr_arr.
            If its values is 0 or False it will return a point for only the unique input scores,
            i.e. numpy.unique(scr_arr) returning values.

    Output:

        tp_rate: True positive rate values array.
        tf_rate: False positive rate values array.
        unique_scores: Unique Scores from scr_arr argument. These values are the thresholds 
            where new points in ROC curve where added.

    """

    #Weak checking for invalid values in real classes (values) array.
    if not (np.sum(trh_arr == 0) + np.sum(trh_arr == 1)) == trh_arr.size and\
        not (np.sum(trh_arr == -1) + np.sum(trh_arr == 1)) == trh_arr.size:
        raise Exception("Samples truth table array contains invalid values:\
                    Only {1,0} or {1,-1} (float or integer) sets of values are valid.")

    #Checking the expected values (True/False) for 'full_curve' argument
    if not isinstance(full_curve, bool) and not full_curve in [0,1]:
        raise Exception("full_curve agument expected values are: Only {0,1} or {True,False}.") 

    #In place convertion of numerical type in case the input is in integer. 
    trh_arr = trh_arr.astype(arr_type, copy=False)
    scr_arr = scr_arr.astype(arr_type, copy=False)

    #Convert truth binary array from {1, -1} to {1, 0}.
    trh_arr[ np.where(trh_arr == -1) ] = 0

    #Counting the total number of positive and negative samples.
    pos_sum = trh_arr.sum()
    neg_sum = trh_arr.size - pos_sum

    #Initialising True Positive and False Positive counters.
    tp = 0
    fp = 0

    #Initialising True Positive and False Positive rates.
    tp_rate = list()
    fp_rate = list()    

    #Initialising last score.
    last_scr = -1
    
    #Building the ROC curve
    for exp_y, scr in zip(trh_arr, scr_arr):

        if exp_y > 0:   #if expected y is 1  
            tp += 1
        else:           #if expected y is 0  
            fp += 1

        if scr != last_scr or full_curve:
            tp_rate.append( tp / pos_sum )
            fp_rate.append( fp / neg_sum )
            last_scr = scr
        
    #Append last point if not already
    tp_rate.append( tp / pos_sum )
    fp_rate.append( fp / neg_sum )

    #Converting TP-Rate and FP-Rate lists to numpy.arrays
    tp_rate = np.array(tp_rate, dtype=arr_type)
    fp_rate = np.array(fp_rate, dtype=arr_type)

    #Returning the ROC curve
    return tp_rate, fp_rate, np.unique(scr_arr)


def pr_curve(trh_arr, scr_arr, full_curve=False, arr_type=np.float32):
    """Precision-Recall (PR) curves.

    Returns the Precision-Recall curve given the truth table, i.e. the binary real 
    labels of the samples and the scores/probabilities returned by the classifier, 
    either as final result or in an indeterminate stage of the classification/prediction.

    This algorithm has been developed for exploiting the monotonicity of the curve, 
    i.e. any instance that is classified positive with respect to a given threshold 
    will be classified positive for all lower thresholds as well. Therefore, we can 
    simply sort the test instances decreasing by f scores and move down the list, 
    processing one instance at a time and updating TP as we go. In this way
    an PR graph can be created from a linear scan. Followin the same line of thought
    as the algorithm is described for ROC curves which has been cited in "ROC Graphs: 
    Notes and Practical Considerations for Researchers" by Tom Fawcett.
    
    Input argumens:
        
        trh_arr: The binay truth array, i.e. the real classies of the samples 
            has been given to the Classifier.
            Valid values:  +1 for positive samples.
                            0 or -1 for negative samples.
            arr_type: (optional) user-defined arrays type. default numpy.flaot32

        scr_arr: The Classifier's returning scores/probabilities array for samples 
            being positive.

        full_curve: Expected values are {0,1} or {True, False}:
            If its values is 1 or True it will return a point for every input score in scr_arr.
            If its values is 0 or False it will return a point for only the unique input scores,
            i.e. numpy.unique(scr_arr) returning values.

    Output:

        precision: Precision values array.
        recall: Recall values array (equivalent to the True positive rate) .
        tp_rate: False positive rate values array.
        unique_scores: Unique Scores from scr_arr argument. These values are the thresholds 
            where new points in PR curve where added.

    """

    #Checking for invalid values in real classes (values) array.
    if not (np.sum(trh_arr == 0) + np.sum(trh_arr == 1)) == trh_arr.size and\
        not (np.sum(trh_arr == -1) + np.sum(trh_arr == 1)) == trh_arr.size:
        raise Exception("Samples truth table array contains invalid values:\
                    Only {1,0} or {1,-1} (float or integer) sets of values are valid.")

    #Checking the expected values (True/False) for 'full_curve' argument
    if not isinstance(full_curve, bool) and not full_curve in [0,1]:
        raise Exception("full_curve agument expected values are: Only {0,1} or {True,False}.") 

    #In place convertion of numerical type in case the input is in integer. 
    trh_arr = trh_arr.astype(arr_type, copy=False)
    scr_arr = scr_arr.astype(arr_type, copy=False)

    #Convert truth binary array from {1, -1} to {1, 0}.
    trh_arr[ np.where(trh_arr == -1) ] = 0

    #Counting the total number of positive.
    pos_sum = trh_arr.sum()

    #Initialising True Positive and False Positive counters.
    tp = 0
   
    #Initialising True Positive and False Positive rates.
    precision = list()
    recall = list()
    
    #Appeding fist fixed point (x, y) = (0, 1).
    #This point might be a duplicate in best case or misleading in worst case.
    precision.append( 1.0 )
    recall.append( 0.0 )
    
    #Initialising last score and document counter.
    last_scr = -1
    doc_cnt = 0.0
    
    #Building the PR curve
    for exp_y, scr in zip(trh_arr, scr_arr):

        doc_cnt += 1.0

        if exp_y > 0:   #if expected y is 1  
            tp += 1

        if scr != last_scr or full_curve:
            precision.append( tp / doc_cnt )
            recall.append( tp / pos_sum )
            last_scr = scr
        
    #Append last point if not already
    precision.append( tp / doc_cnt )
    recall.append( tp / pos_sum )

    #Converting Precision and Recall lists to numpy.arrays
    precision = np.array(precision, dtype=arr_type)
    recall = np.array(recall, dtype=arr_type)

    #Returning the ROC curve
    return precision, recall, np.unique(scr_arr)


def auc(x, y, arr_type=np.float32):
    """Area Under the Curve (AUC).

    Returns the Area Under the Curve of a given curve (ROC, PR, etc.).
    The trapezoid approximation is used as instructed in several tutorial 
    and papers such as "ROC Graphs: Notes and Practical Considerations for 
    Researchers" by Tom Fawcett.

    It performed an ascending order checking on curve's coordinates sequences assuring that
    the results will always be correct. In case the order is not in ascending order the points 
    are reordered.

    Input arguments:

        x: is the numpy.array sequence of all x coodinates of the curve's points.
        y: is the numpy.array sequence of all y coodinates of the curve's points.
        arr_type: (optional) user-defined arrays type. default numpy.flaot32

    Output:

        auc: a floating point value equal to the sum of trapezoids formed
            by the curve's points given as aruments.

    """

    #Checking the validity of the x and y arrays before start computing the AUC.
    if x.size != y.size:
        raise Exception("X and Y coordinate arrays must have equal length.")

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise Exception("Invalid array-argument's Dimensions: Only 1D arrays are acceptable\
                    for this implementation of computing AUC.")
    
    if x.size < 2:
        raise ValueError("AUC cannot computed given a single point.")

    #In place convertion of numerical type in case the input is in integer. 
    x = x.astype(arr_type, copy=False)
    y = y.astype(arr_type, copy=False)

    #Checking curve's points correct sequence.
    if np.argmax(x) == 0: #Sequence assumed to be in descending order so it is inverted.
        y = y[::-1]
        x = x[::-1]

    #Checking the Sequence. If x it is in random order the input is not a proper curve.
    if not np.array_equal(np.sort(x), x) and not np.array_equal(np.sort(x), x[::-1]): 
        raise Exception("X-coordinate sequence is in random oder: Impossible to calculate the AUC correctly.")
    
    #Calculating the delta-x's, i.e. bases of trapezoids.
    dx = np.abs( np.diff(x) )
    
    #Calculating the average hight amongst all sequential pairs of y axis heights.
    height_means = ( y[1:] + y[:-1] ) / 2.0 

    #Returning the AUC, i.e. the sum of all trapezoids formed under the curve.
    #This is equal to matrix operation h*b.T or array operation sum(h*b).
    return np.sum( height_means*dx )


def smooth_linear(y, x=None, arr_type=np.float32):
    """Practical linear smoothing function. 

    Takes as argument a saw-like shaped curve and returns the curve connecting the local maxima of the 
    curve. There is no need for X axis values to be given in case only the Y axis values are required 
    for all x values.

    In case the x values are provided then only the y thresholds (local maxima) with their respective
    x values will be retunred.   

    NOTE: ***The Y sequence sould be inverted from the lowest to the highest values ***
          *** I works properly for curves that increasing or decreasing for all x values 
              (Think about it by studying the code bellow)***

     Input arguments:

        y: is the numpy.array sequence of all y coodinates of the curve's points.
        x: is the numpy.array sequence of all x coodinates of the curve's points.
        arr_type: (optional) user-defined arrays type. default numpy.flaot32

    Output:

        smooth_y: is the numpy.array sequence of all local maxima of y axis.
        smooth_x: is the numpy.array sequence of the respective x of smooth_y sequence. 
                  *(only when x values sequence is provided as input argument.)

    """

    #The list of local maxima. Either for each x value or only for the unique local maxima.
    smooth_y = list()
    
    #Finding the unique local maxima. 
    #The y values sequence assumed to be given from lower to higher value
    max_y = 0
    for yval in y:
       
        #The same local maxima is appended to the list until a greater value is occuring.
        if yval > max_y:
            max_y = yval

        smooth_y.append( max_y )

    #Converting the list to an numpy array using the proper data type given as argument.
    smooth_y = np.array(y, dtype=arr_type)

    #Finding the thresholds and their respective x values in case x sequence is given as argument.
    if isinstance(x, np.ndarray) or isinstance(x, list):
        
        #Getting the unique maximum values.
        y_thresholds, u_inds = np.unique( smooth_y, return_index=True )
        
        #Getting the respective x for each threashold.
        smooth_x = np.array(x, dtype=arr_type)[u_inds]

        #Retunring the values. 
        return y_thresholds, smooth_x

    #Returning the smoothed y values sequence where the values between the local maxima 
    #has been replaced with the local maxima values, whenever a new one has been occuring
    return smooth_y


""" Under Refactoring - START"""

def recl_lv_P_avg(P, R):

    """Recall Level Precision Averages
    This function is smoothing out Precision-Recall Curve
    via interpolation. That is, always 

    """

    #Creating the array of 11 Recall Levels (0 to 10)
    R_Levels = np.arange(0.0, 1.1, 0.1)
    
    #Smoothing out the Precision (Y axis) of the P-R Curve
    #CRITICAL: The P sould be inverted from the lowest to 
    #the highest values*. 
    # *( it suppose the higest values to be normally fist in order )

    smthd_P = smooth_linear(P[::-1])

    smthd_P = smthd_P[::-1]
    
    SP = np.zeros_like(R_Levels, dtype=np.float)

    down_r_level_idxs = 0
    for i, r in enumerate(R_Levels):
        
        up_r_level_idxs = np.max( np.where(R <= r) )
        
        print up_r_level_idxs

        #************************** ITS GETTING BETTER NEEDS MORE WORK TO CLEAN-UP THE MESS **********************
        
        #print smthd_P[up2_r_level_idxs]
        #print np.sum( smthd_P[up2_r_level_idxs] )
        #print float( len(up2_r_level_idxs) )

        SP[i] = np.mean( smthd_P[ down_r_level_idxs : up_r_level_idxs ] )

        down_r_level_idxs = up_r_level_idxs
                 
    return SP, R_Levels


def close_to_11_rl(P, R): 
    
    #Creating the array of 11 Recall Levels (0 to 10)
    R_Levels = np.arange(0.0, 1.1, 0.1)
    
    #Smoothing out the Precision (Y axis) of the P-R Curve
    #CRITICAL: The P sould be inverted from the lowest to 
    #the highest values*. 
    # *( it suppose the higest values to be normally fist in order )
    smthd_P = smooth_linear(P[::-1])

    smthd_P = smthd_P[::-1]
    
    idxs = np.zeros_like(R_Levels, dtype=np.int)
    
    for i, r in enumerate(R_Levels[1::]):
        lft_idx = np.max(np.where(R < r))
        rgt_idx = np.max(np.where(R >= r))
        
        #print smthd_P[lft_idx], ">", smthd_P[rgt_idx]
        if smthd_P[lft_idx] >= smthd_P[rgt_idx]:
            idxs[i+1] = lft_idx
        elif smthd_P[lft_idx] < smthd_P[rgt_idx]:
            idxs[i+1] = rgt_idx
            
        #print lft_idx, rgt_idx, idxs[i+1] 
    #idxs[0] = 0
    #idxs[10] = smthd_P
    
    SP = smthd_P[idxs]
    #SP[10] = 0
    
    #print R_Levels
    #print SP
             
    return SP, R_Levels


def std_avg_pr_curve(P, R):
    ### IT SEEMS CRAP
    
    R_Levels = np.arange(0.0, 1.1, 0.1) 
    
    ipol_P = np.zeros_like(P, dtype=np.float64)
    
    max_p = 0
    for i, p in enumerate(P[::-1]):
        if p > max_p:
            max_p = p
        ipol_P[i] = max_p     
    
    P_AVG = np.zeros(11, dtype=np.float64)
    
    for i, r in enumerate(R_Levels):
        P_AVG[i] = np.average(ipol_P[np.where(R <= r)])
    
    return P_AVG, R_Levels 


def nrst_smooth_pr(P, R):
    
    R_Levels = np.arange(0.0, 1.1, 0.1)
    
    smoothed_P = smooth_linear(P[::-1])

    smoothed_P = smoothed_P[::-1]
    
    idxs = np.zeros_like(R_Levels, dtype=np.int)
    for i, r in enumerate(R_Levels):
        idxs[i] = (np.abs(R - r)).argmin()
    
    return smoothed_P[idxs], R[idxs]


""" Under Refactoring - END"""


class purepy(object):

    @staticmethod
    def roc_curve(truth_d, scr_d):
        """Receiver Operating Characteristic (ROC)"""

        pos_sum = sum( [y=='Y' for y in truth_d.values()] )
        neg_sum = len(truth_d) - pos_sum

        pos_cnt = 0
        neg_cnt = 0

        scr_ybin_lst = list()
        for key, bin_val in truth_d.items():

            if key in scr_d:
                bin_int = 1 if bin_val == 'Y' else 0
                scr_ybin_lst.append( (scr_d[key], bin_int) )
            else:
                #add no provided aswares with negative value in Ground-Truth file
                neg_cnt += 1 if bin_val == 'N' else 0

        scr_ybin_srd_lst = sorted(scr_ybin_lst, key=lambda scr_ybin_lst: scr_ybin_lst[0], reverse=True)

        tp_rate = list()
        fp_rate = list()    

        last_scr = -1
        #append [0, neg_cnt / float(neg_sum)]
        #tp_rate.append( 0.0 )
        #fp_rate.append( neg_cnt / float(neg_sum) )

        for i, (scr, y) in enumerate(scr_ybin_srd_lst):

            if scr != last_scr:
                tp_rate.append( pos_cnt / float(pos_sum) )
                fp_rate.append( neg_cnt / float(neg_sum) )
                last_scr = scr

            if int(y) == 1:
                pos_cnt += 1
            elif int(y) == 0 or int(y) == -1:
                neg_cnt += 1
            else:
                raise Exception("Incompatible input: -1, 0, 1 values supported for Y binary")
        
        #Append last point if not already
        tp_rate.append( pos_cnt / float(pos_sum) )
        fp_rate.append( neg_cnt / float(neg_sum) )

        norm = 1.0 #float(pos_sum*neg_sum)

        return tp_rate, fp_rate, norm 

    @staticmethod
    def auc(roc_curve):

        if isinstance(roc_curve, str):
            roc_curve = list(eval(roc_curve))

        y = roc_curve[0]
        x = roc_curve[1]

        dx = [float(x1 - x2) for x1, x2 in zip(x[1::],x)]
        dx0 = [0]
        dx0.extend(dx)

        h = [(y1 + y2)/2.0 for y1, y2 in zip(y[1::],y)]
        h0 = [0]
        h0.extend(h)

        return sum( [dx*y for dx, y in zip(dx0, h0)] ) / roc_curve[2]


