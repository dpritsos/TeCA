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

genres = [ "news" , "product_companies", "forum", "blogs", "wiki_pages" ] #academic
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
    resaults.append( pool.dispatch( vg.ngrams_vects_from_path, g, base_filepath, ng_size=3, multiproc=False) )
    #print("Experiment %s VS All: Dispatched" % g)
    
for res in resaults:
    genre, global_ngram_dict, webpg_l, ngram_vect_l = res.value 
    print(len(webpg_l), len(ngram_vect_l), len(ngram_vect_l))
    #global_ngram_dict = gterm_d_gen(ngram_vect_l)  
    #Save Term Frequency Dictionaries 
    #filename = g  + ".corpd"   
    #if save_dct(filename, global_term_dict, (filepath + "corpus_dictionaries/") ):
    #    print( g + " CORPUS TERMS DICTIONARY: SAVED!" )
    #filename = g + ".cvect"
    #if save_dct_lst( filename, webpg_vect_l, webpg_l, (filepath + "corpus_webpage_vectors/") ):
    #    print( g + " CORPUS VECTORS: SAVED!" )
    #Save Ngram Frequency Dictionaries
    filename = genre  + ".ncorpd"   
    if save_dct(filename, global_ngram_dict, (filepath + "ngrams_corpus_dictionaries/") ):
        print( genre + " NGRAMS CORPUS TERMS DICTIONARY: SAVED!" )
    filename = genre + ".ncvect"
    if save_dct_lst( filename, ngram_vect_l, webpg_l, (filepath + "ngrams_corpus_webpage_vectors/") ):
        print( genre + " NGRAMS CORPUS VECTORS: SAVED!" )    

pool.join_all()

print "Thank you and Goodbye!"

