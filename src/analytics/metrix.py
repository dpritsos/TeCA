""" 
    This module includs a set of functions required for interpolating and averaging
    PR curves as they are returned from 'sklearn.metrics' machine learning library.

    Aurthor: Dimitrios Pritos

"""

import numpy as np


def roc_curve(trh_arr, scr_arr, arr_type=np.float32):
    """Receiver Operating Characteristic (ROC) curves.

    Returns the Receiver Operating Characteristic curve given the truth table, 
    i.e. the binary real labels of the samples and the scores/probabilities returned. 
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

    Output:

        tp_rate: True positive rate values array.
        tp_rate: False positive rate values array.
        unique_scores: Unique Scores from scr_arr argument. These values are the thresholds 
            where new points in ROC curve where added.

    """

    #Weak checking for invalid values in real classes (values) array.
    if not (np.sum(trh_arr == 0) + np.sum(trh_arr == 1)) == trh_arr.size and\
        not (np.sum(trh_arr == -1) + np.sum(trh_arr == 1)) == trh_arr.size:
        raise Exception("Samples truth table array contains invalid values:\
                    Only {1,0} or {1,-1} (float or integer) sets of values are valid.")

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

        if scr != last_scr:
            tp_rate.append( tp / pos_sum )
            fp_rate.append( fp / neg_sum )
            last_scr = scr

        if exp_y > 0:   #if expected y is 1  
            tp += 1
        else:           #if expected y is 0  
            fp += 1
        
    #Append last point if not already
    tp_rate.append( tp / pos_sum )
    fp_rate.append( fp / neg_sum )

    #Converting TP-Rate and FP-Rate lists to numpy.arrays
    tp_rate = np.array(tp_rate, dtype=arr_type)
    fp_rate = np.array(fp_rate, dtype=arr_type)

    #Returning the ROC curve
    return tp_rate, fp_rate, np.unique(scr_arr)


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

    #Checking curve's points correct sequence. Channing it if it is required.
    if np.argmax(x) == 0: #Sequence is in descending order so it is inverted.
        y = y[::-1]
        x = x[::-1]
    elif np.argmin(x) != 0: #Sequence is not in random order the the curve is incorrect.
        raise Exception("X-coordinate sequence is in random oder: Impossible to calculate the AUC correctly.")
    
    #Calculating the delta-x's, i.e. bases of trapezoids.
    dx = np.diff(x)
    
    #Calculating the average hight amongst all sequential pairs of y axis heights.
    height_means = ( y[1:] + y[:-1] ) / 2.0 

    #Returning the AUC, i.e. the sum of all trapezoids formed under the curve.
    #This is equal to matrix operation h*b.T or array operation sum(h*b).
    return np.sum( heigt_means*dx )


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
    smthd_P = interpol_linear(P)

    smthd_P = smthd_P[::-1]
    
    SP = np.zeros_like(R_Levels, dtype=np.float)

    for i, r in enumerate(R_Levels):
        
        up2_r_level_idxs = np.where(R <= r)
        print r
        print up2_r_level_idxs
        
        print smthd_P[up2_r_level_idxs]
        print np.sum( smthd_P[up2_r_level_idxs] )
        print float( len(up2_r_level_idxs) )

        SP[i] = np.mean( smthd_P[up2_r_level_idxs] )
            
        #print lft_idx, rgt_idx, idxs[i+1] 
    #idxs[0] = 0
    #idxs[10] = smthd_P
    
    
    #SP[10] = 0
    
    #print R_Levels
    #print SP
             
    return SP, R_Levels


def close_to_11_rl(P, R):
    
    #Creating the array of 11 Recall Levels (0 to 10)
    R_Levels = np.arange(0.0, 1.1, 0.1)
    
    #Smoothing out the Precision (Y axis) of the P-R Curve
    #CRITICAL: The P sould be inverted from the lowest to 
    #the highest values*. 
    # *( it suppose the higest values to be normally fist in order )
    smthd_P = p_smooth(P)

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


def interpol_linear(P):
    
    ipol_P = np.zeros_like(P, dtype=np.float64)
    
    max_p = 0
    for i, p in enumerate(P):
       
        if p > max_p:
            max_p = p
       
        ipol_P[i] = max_p     
    
    return ipol_P


def std_avg_pr_curve(P, R):
    
    R_Levels = np.arange(0.0, 1.1, 0.1) 
    
    ipol_P = np.zeros_like(P, dtype=np.float64)
    
    max_p = 0
    for i, p in enumerate(P):
        if p > max_p:
            max_p = p
        ipol_P[i] = max_p     
    
    P_AVG = np.zeros(11, dtype=np.float64)
    
    for i, r in enumerate(R_Levels):
        P_AVG[i] = np.average(ipol_P[np.where(R <= r)])
    
    return R_Levels, P_AVG


def nrst_smooth_pr(P, R):
    
    R_Levels = np.arange(0.0, 1.1, 0.1)
    
    smoothed_P = smooth_pr_curve(P)
    
    idxs = np.zeros_like(R_Levels, dtype=np.int)
    for i, r in enumerate(R_Levels):
        idxs[i] = (np.abs(R - r)).argmin()
    
    return smoothed_P[idxs], R[idxs]


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

