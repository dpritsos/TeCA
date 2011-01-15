"""

"""
import os
from vectorhandlingtools import *
import termvectorgenerator as tvg
from trainevaloneclssvm import *

genres = [ "news", "product_companies", "forum", "blogs", "wiki_pages" ] #academic "news"
base_filepath = "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/" 
ng_vectors_d = "/ngrams_corpus_webpage_vectors/"

vform = TermVectorFormat()

resaults = list()
for g in genres:
    filepath = str( base_filepath + g + "/" )
    if filepath and not os.path.isdir((filepath + "ngrams_corpus_dictionaries/")):
        os.mkdir((filepath + "ngrams_corpus_dictionaries/"))
    wpsl_genre = dict() 
    ng_vectl_genre = dict()
    vform.glob_vect_l(base_filepath, ng_vectors_d, [ g ] , None, None, wpsl_genre, ng_vectl_genre, True)
    print ng_vectl_genre[ g ][0]
    global_ngram_dict = gterm_d_gen(ng_vectl_genre[ g ]) 
    #Save Ngram Frequency Dictionaries
    filename = g  + ".ncorpd"   
    if save_dct(filename, global_ngram_dict, (filepath + "ngrams_corpus_dictionaries/") ):
        print( g + " NGRAMS CORPUS TERMS DICTIONARY: SAVED!" )
    wpsl_genre = None 
    ng_vectl_genre = None
            
print "Thank you and Goodbye!"

