
import tables as tb
import numpy as np

hfile = tb.openFile("/home/dimitrios/Synergy-Crawler/Santini_corpus/Santini_corpus.h5")

for tb in hfile.walkNodes("/Santini_corpus", classname="Table"):
    if tb.name != "PageListTable":
        recarr = tb.read()
        #print recarr.dtype
        #print recarr
        for rec in recarr:
            print rec
            if len(rec[0]) < 3:
                0/0    
    #print tb.read() 

#print hfile


hfile.close()