
import tables as tb
import numpy as np

#[ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]

fh = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/CSVM_CrossVal_Results.h5', 'r')

print fh
#arr = fh.root.Res_eshop_vs_rest.where('nu < 0.8')
arr = fh.root.MultiClass_CrossVal
c = np.array([0.0], dtype=np.float32)
print arr.read()
print np.sum(arr.read()['Acc'])/10
    #if rr[1] > 0.2 and rr[1] < 0.8:
    #    print rr[0:6]
    #    c += rr[3]

#print c
#print c/10

        
#print np.where(arr['F1'] > 0.0)

fh.close()