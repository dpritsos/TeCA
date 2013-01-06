
import numpy as np

def STD_AVG_PR(P, R):
    
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


def SMOOTH_PR(P):
    
    ipol_P = np.zeros_like(P, dtype=np.float64)
    
    max_p = 0
    for i, p in enumerate(P):
        if p > max_p:
            max_p = p
        ipol_P[i] = max_p     
    
    return ipol_P
