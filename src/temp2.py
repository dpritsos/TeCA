
import tables as tb
import numpy as np

#[ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]

fh = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/3grams-OCSVM_CrossVal_Results_NORMALIZED.h5', 'r')
#fh = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/CSVM_CrossVal_Results.h5', 'r')


print fh
#print fh.root.spage_MultiClass_CrossVal.read()

 
for tbpntr in fh.walkNodes('/', classname='Table'):
    tb = tbpntr.read()
    print tbpntr.name
    for nu in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.80, 0.9, 0.95]:
        c = 0.0
        for row in tb[ np.where( (tb['nu'] == nu) & (tb['feat_num'] == 30)) ]:
            #print row 
            c += row[3]
        print c/10.0, nu
    print "\n" 
#arr = fh.root.Res_eshop_vs_rest.where('nu < 0.8')
#arr = fh.root.
#c = np.array([0.0], dtype=np.float32)
#print arr.read()
#print np.sum(arr.read()['Acc'])/10
    #if rr[1] > 0.2 and rr[1] < 0.8:
    #    print rr[0:6]
    #    c += rr[3]

#print c
#print c/10

        
#print np.where(arr['F1'] > 0.0)

fh.close()