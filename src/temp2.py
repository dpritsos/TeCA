
import tables as tb
import numpy as np

#[ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]

fh = tb.openFile('/home/dimitrios/Synergy-Crawler/Santini_corpus/Res_10FldC_3grams_Freq_OCSVM.h5', 'r')

print fh
#arr = fh.root.Res_eshop_vs_rest.where('nu < 0.8')
arr = fh.root.Res_spage_vs_rest.iterrows()
c = np.array([0.0], dtype=np.float32)
for rr in arr:
    if rr[1] > 0.2 and rr[1] < 0.8:
        print rr[0:6]
        c += rr[3]

print c
print c/10

        
#print np.where(arr['F1'] > 0.0)

fh.close()