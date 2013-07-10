
import numpy as np

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


def smooth_pr_curve(P):
    
    ipol_P = np.zeros_like(P, dtype=np.float64)
    
    max_p = 0
    for i, p in enumerate(P):
        if p > max_p:
            max_p = p
        ipol_P[i] = max_p     
    
    return ipol_P


def nrst_smooth_pr(P, R):
    
    R_Levels = np.arange(0.0, 1.1, 0.1)
    
    smoothed_P = smooth_pr_curve(P)
    
    idxs = np.zeros_like(R_Levels, dtype=np.int)
    for i, r in enumerate(R_Levels):
        idxs[i] = (np.abs(R - r)).argmin()
    
    return smoothed_P[idxs], R[idxs]


def interpol_soothed_pr_curve(P, R):
    
    R_Levels = np.arange(0.0, 1.1, 0.1)
    
    smthd_P = smooth_pr_curve(P)
    #SOS
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
