
import numpy as np
import scipy.sparse as ssp

trn_idxs = [2,3,11,14]
cls_gnr_tgs = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2]

corpus_mtrx = np.array([[1,2,34,5,0,0,0,0,6,7,9,9],\
                [1,2,34,5,6,7,9,0,0,0,0,9],\
                [2,2,0,0,0,0,34,5,6,7,9,9],\
                [3,2,34,5,6,7,0,0,0,0,9,9],\
                [4,2,34,5,6,0,0,0,0,7,9,9],\
                [5,0,0,0,0,2,34,5,6,7,9,9],\
                [6,2,34,5,0,0,0,0,6,7,9,9],\
                [7,0,0,0,0,2,34,5,6,7,9,9],\
                [8,2,34,5,6,7,9,0,0,0,0,9],\
                [9,2,34,5,6,7,9,0,0,0,0,9],\
                [10,2,34,5,6,7,0,0,0,0,9,9],\
                [11,2,34,5,6,0,0,0,0,7,9,9],\
                [12,2,34,5,0,0,0,0,6,7,9,9],\
                [13,2,0,0,0,0,34,5,6,7,9,9],\
                [14,2,34,5,0,0,0,0,6,7,9,9]])


genres_lst = ['test1', 'test2']




def contruct_classes(trn_idxs, corpus_mtrx, cls_gnr_tgs):
    inds_per_gnr = dict()
    inds = list()
    last_gnr_tag = 1
    
    for trn_idx in trn_idxs:
        
        if cls_gnr_tgs[trn_idx] != last_gnr_tag:
            print genres_lst[last_gnr_tag - 1]
            inds_per_gnr[ genres_lst[last_gnr_tag - 1] ] = inds
            print inds_per_gnr[ genres_lst[last_gnr_tag - 1] ]
            last_gnr_tag = cls_gnr_tgs[trn_idx]
            inds = []
        
        inds.append( trn_idx )    
    
    print genres_lst[last_gnr_tag - 1]
    inds_per_gnr[ genres_lst[last_gnr_tag - 1] ] = inds
    print inds_per_gnr[ genres_lst[last_gnr_tag - 1] ] 

    gnr_classes = dict()
    for g in genres_lst:
        #Merge All Term-Frequency Dictionaries created by the Raw Texts
        print g
        print inds_per_gnr[g]
        print corpus_mtrx[inds_per_gnr[g], :]
        gnr_classes[g] = corpus_mtrx[inds_per_gnr[g], :].sum(axis=0)
    
    return (gnr_classes, inds_per_gnr) 

print contruct_classes(trn_idxs, corpus_mtrx, cls_gnr_tgs)