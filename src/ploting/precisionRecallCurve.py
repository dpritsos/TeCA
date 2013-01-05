" "

import numpy as np
import PR_curves_to_11_standard_recall_levels as srl

def prcurve_sums_bin(ranked_predicted_y):
    
    rpy = ranked_predicted_y
    
    total_pos = np.sum(rpy == 1)
    zeros = np.where(rpy == 0)
    zeros_idx = zeros[0]
    #print zeros_idx
    
    P = np.zeros_like(zeros_idx, dtype=np.float64)
    R = np.zeros_like(zeros_idx, dtype=np.float64)
    
    for i, poss in enumerate(zeros_idx):
        #print (poss - i),"/",float(total_pos)
        R[i] = (poss - i)/float(total_pos)
        P[i] = (poss - i)/float(poss)
        
    return P, R


def prcurve_bin(ranked_predicted_y):
    
    rpy = ranked_predicted_y
    
    total_pos = np.sum(rpy == 1)
    
    P = np.zeros_like(rpy, dtype=np.float64)
    R = np.zeros_like(rpy, dtype=np.float64)
    
    zero_c = 0
    for i, bin_val in enumerate(rpy):
        if bin_val == 0:
            zero_c += 1
        print i - zero_c,"/",float(total_pos)
        R[i] = (i - zero_c) / float(total_pos)
        if i:
            P[i] = (i - zero_c) / float(i)
        
    return P, R


if __name__ == '__main__':
    test = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\
                     1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int)
    
    
    P, R = prcurve_bin(test)
    
    print srl.STD_AVG_PR(P, R)
    
         

        
         
    