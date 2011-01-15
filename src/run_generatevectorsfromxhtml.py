"""

"""
import os
from vectorhandlingtools import *
import termvectorgenerator as tvg
import sys
sys.path.append('../../synergeticprocessing/src')
from synergeticpool import *

pool = SynergeticPool( { '192.168.1.65':(40000,'123456'), '192.168.1.68':(40000,'123456') }, local_workers=1, syn_listener_port=41000 ) 
print "Registering"
pool.register_mod( ['termvectorgenerator', 'vectorhandlingtools'] )  
print "Regitered OK"

genres = [ "news", "product_companies", "forum", "blogs", "wiki_pages" ] #academic "news"
base_filepath = ["/home/dimitrios/Documents/Synergy-Crawler/saved_pages/", "../Documents/Synergy-Crawler/saved_pages/"] 

vg = tvg.VectGen()

resaults = list()
for g in genres:
    filepath = str( base_filepath[0] + g + "/" )
    if filepath and not os.path.isdir((filepath + "corpus_dictionaries/")):
        os.mkdir((filepath + "corpus_dictionaries/"))
    if filepath and not os.path.isdir((filepath + "corpus_webpage_vectors/")):
        os.mkdir((filepath + "corpus_webpage_vectors/"))
    if filepath and not os.path.isdir((filepath + "ngrams_corpus_dictionaries/")):
        os.mkdir((filepath + "ngrams_corpus_dictionaries/"))
    if filepath and not os.path.isdir((filepath + "ngrams_corpus_webpage_vectors/")):
        os.mkdir((filepath + "ngrams_corpus_webpage_vectors/"))
    #
    filepath = base_filepath[0] + g  
    xhtmlfiles_l = [files for path, dirs, files in os.walk(filepath)]
    if xhtmlfiles_l:
        xhtmlfiles_l = xhtmlfiles_l[0]
    count = 0
    xfl_extended =  [ (file, g, base_filepath, 3) for file in xhtmlfiles_l ]
    print xfl_extended[0]
    for res in pool.imap( vg.ngrams_vects_from_file, xfl_extended ):
    
    #for file in xhtmlfiles_l:
    #    print "Dispatched: ", file
    #    res = pool.dispatch( vg.ngrams_vects_from_file, file, g, base_filepath, ng_size=3)
    #    resaults.append( res.value )
    #    count += 1 
    #    print "Appended", count 
    #ngram_vect_l = list()
    #webpg_l = list()    
    #for res in resaults:
        genre, webpg, ngram_vect = res
        
        if genre != g:
            raise Exception('RESAULT ERROR')    
        if ngram_vect:
             
            #ngram_vect_l.append(ngram_vect)
            #webpg_l.append(webpg)
    #global_ngram_dict = gterm_d_gen(ngram_vect_l) 
    #print(len(webpg_l), len(ngram_vect_l), len(global_ngram_dict))
    #global_ngram_dict = gterm_d_gen(ngram_vect_l)  
    #Save Term Frequency Dictionaries 
    #filename = g  + ".corpd"   
    #if save_dct(filename, global_term_dict, (filepath + "corpus_dictionaries/") ):
    #    print( g + " CORPUS TERMS DICTIONARY: SAVED!" )
    #filename = g + ".cvect"
    #if save_dct_lst( filename, webpg_vect_l, webpg_l, (filepath + "corpus_webpage_vectors/") ):
    #    print( g + " CORPUS VECTORS: SAVED!" )
    #Save Ngram Frequency Dictionaries
        #filename = g  + ".ncorpd"   
        #if save_dct(filename, global_ngram_dict, (filepath + "ngrams_corpus_dictionaries/") ):
        #    print( g + " NGRAMS CORPUS TERMS DICTIONARY: SAVED!" )
            filename = webpg + "." + g + ".ncvect"
            #if save_dct_lst( filename, ngram_vect_l, webpg_l, (filepath + "ngrams_corpus_webpage_vectors/") ):
            if save_dct_lst( filename, [ngram_vect], [webpg], (filepath + "/ngrams_corpus_webpage_vectors/") ):
                print( g + " NGRAMS CORPUS VECTORS: SAVED!" )    

pool.join_all()

print "Thank you and Goodbye!"

