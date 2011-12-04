
import sys
sys.path.append('../../html2vectors/src')
import tables as tb
import matplotlib.pyplot as matplt
import numpy as np
import html2tf.tables.tbtools as tbtls
import html2tf.dictionaries.tfdtools as tdtls

tfdhdlr = tdtls.TFdictHandler()

fileh = tb.openFile('/home/dimitrios/Synergy-Crawler/saved_pages/ACC.h5', 'r')
#fileh = tb.openFile('/home/dimitrios/Synergy-Crawler/Santini_corpus/Santini_corpus.h5', 'r+')
#genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
genres = [ "blogs", "forum", "news", "product_companies", "wiki_pages" ] 
#pg_lst_tb = fileh.root.AutomatedCollected_corpus.trigrams.wiki_pages.PageListTable

#c =0 
#for row in pg_lst_tb:
#    if row['status_code']==1:
#        c+=1
#        #print row['terms_num']
#print c 


dictionarys = list()

tb_not = ["PageListTable", "Dictionary"]
for g in genres:
    print g
    for tf_tb in fileh.walkNodes("/AutomatedCollected_corpus/trigrams/" + g, classname='Table'):
        if tf_tb.name == tb_not: continue
        dictionarys.append( tf_tb.read() )

for g in genres:
    print g
    tf_tb = fileh.getNode("/AutomatedCollected_corpus/trigrams/" + g, "/DictionaryNormMax")
    dictionarys.append( tf_tb.read() )
    
globalDi = fileh.root.AutomatedCollected_corpus.trigrams.CorpusGlobalDictionaryNormMax.read()
globalDi.sort(order='terms') 

tf_d = dict(list(globalDi))

print "OK"



#dictionarys.append( fileh.root.AutomatedCollected_corpus.trigrams.blogs.DictionaryNormMax.read() ) #B
#dictionarys.append( fileh.root.AutomatedCollected_corpus.trigrams.news.DictionaryNormMax.read() )  #G
#dictionarys.append( fileh.root.AutomatedCollected_corpus.trigrams.forum.DictionaryNormMax.read() ) #R
#dictionarys.append( fileh.root.AutomatedCollected_corpus.trigrams.product_companies.DictionaryNormMax.read() ) #C
#dictionarys.append( fileh.root.AutomatedCollected_corpus.trigrams.wiki_pages.DictionaryNormMax.read() ) #M
for g in genres:
    tf_tb = fileh.getNode("/Santini_corpus/trigrams/" + g, "/DictionaryNormMax")
    dictionarys.append( tf_tb.read() ) #B

globalDi = fileh.root.Santini_corpus.trigrams.CorpusGlobalDictionaryNormMax.read()
globalDi.sort(order='terms') 

#tf_d = dict([(float(hex(trm)),frq) for trm, frq in globalDi])
tf_d = dict(list(globalDi))
#print len(tf_d)
#print len(globalDi)
#ll = tf_d.items()
#ll.sort()
#for i,z in zip(ll, globalDi):
#    print i, z
#0/0
#Du, ix = np.unique1d(globalDi, return_inverse=True)
#print Du
term_idx = tfdhdlr.tf2tidx(tf_d)
#for i in term_idx.items():
#    print i 
#del tf_d
#0/0
print(len(term_idx))
for dictionary in dictionarys:
    dictionary.sort(order='terms')
    di = dictionary
    print(len(di)) 
    idi = np.zeros(len(di), dtype=[('idx', np.int), ('freq', np.float)])
    algnd_idi = np.zeros(len(term_idx), dtype=[('idx', np.int), ('freq', np.float)])
    for ii, irow in enumerate(algnd_idi):
        irow['idx'] = ii
    for i, row in enumerate(di): 
        idi[i] = ( term_idx[ row['terms'] ], row['freq'] )
    print idi[0]      
    for idi_row in idi:
        #if idi_row['idx'] == 163513:
        #    break
        #print idi_row['idx'], idi_row['freq']
        algnd_idi[ idi_row['idx'] - 1 ]['freq'] = idi_row['freq'] 
        print algnd_idi[ idi_row['idx'] - 1 ]
    print algnd_idi
    #print len(di['terms'])
    #print di['terms'][10:]
    algnd_idi.sort(order='freq')
    matplt.plot(algnd_idi['idx'], algnd_idi['freq'], '^')
matplt.show()






#c = 0
#for tb_in_gr in fileh.walkNodes(where="/AutomatedCollected_corpus/trigrams/blogs", classname="Table"):
#    if tb_in_gr.name != "PageListTable":
        #name = ""   
        #tbl = tb_in_gr.read()
        #print tbl['freq']
        #matplt.plot(np.arange(len(tbl['terms'])), tbl['freq'], '^') 
        #for row in tb_in_gr.where('terms == "~:~"'): #iterrows():
        #    print "Print:", row
            
#        if  tb_in_gr.where('terms == ""'):
#            print tb_in_gr.readWhere('terms == ""')
        #else:
        #    print "No"
        #if c == 2000:
        #    break
        #c += 1
            #if row['freq'] > 100000.0:
                #if name != tb_in_gr.name:
            #        print tb_in_gr.name
            #        print row
                    #name = tb_in_gr.name
#matplt.show()
#fileh.close()
"""

#tbt = tbtls.TFTablesHandler()
#genres_diz = [ "/AutomatedCollected_corpus/trigrams/blogs/DictionaryNormMax",\
#               "/AutomatedCollected_corpus/trigrams/forum/DictionaryNormMax",\
#               "/AutomatedCollected_corpus/trigrams/news/DictionaryNormMax",\
#               "/AutomatedCollected_corpus/trigrams/product_companies/DictionaryNormMax",\
#              "/AutomatedCollected_corpus/trigrams/wiki_pages/DictionaryNormMax" ]
#dictionary = tbt.merge_tf_tbls_Dicts(fileh, genres_diz, "/AutomatedCollected_corpus/trigrams/")

#dictionary = tbt.merge_tf_tbls_using_dict_normalized(fileh, "/AutomatedCollected_corpus/trigrams/blogs", tb_not=["PageListTable", "Dictionary"])
#dictionary = tbt.merge_tf_tbls_using_dict_normalized(fileh, "/AutomatedCollected_corpus/trigrams/forum", tb_not=["PageListTable", "Dictionary"])
#dictionary = tbt.merge_tf_tbls_using_dict_normalized(fileh, "/AutomatedCollected_corpus/trigrams/news", tb_not=["PageListTable", "Dictionary"])
#dictionary = tbt.merge_tf_tbls_using_dict_normalized(fileh, "/AutomatedCollected_corpus/trigrams/product_companies", tb_not=["PageListTable", "Dictionary"])
#dictionary = tbt.merge_tf_tbls_using_dict_normalized(fileh, "/AutomatedCollected_corpus/trigrams/wiki_pages", tb_not=["PageListTable", "Dictionary"])

"""
genres_diz = list()
for g in genres:
    genres_diz.append( "/Santini_corpus/trigrams/" + g + "/DictionaryNormMax" ) 
    dictionary = tbt.merge_tf_tbls_using_dict_normalized(fileh, "/Santini_corpus/trigrams/"+g, tb_not=["PageListTable", "Dictionary"])

dictionary = tbt.merge_tf_tbls_Dicts(fileh, genres_diz, "/Santini_corpus/trigrams/")
"""
fileh.close()







