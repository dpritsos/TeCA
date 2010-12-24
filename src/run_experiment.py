"""

"""
import os
from vectorhandlingtools import *
import sys
sys.path.append('../../synergeticprocessing/src')
from synergeticpool import *
from experiments import *



lower_case = True

##################### CREAT GLOBAL INDEX FOR BOTH CORPUSSES ##################
genres = [ "news" , "product_companies", "forum", "blogs", "wiki_pages"] # "academic", 
base_filepath = "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/"
corpus_d = "/corpus_dictionaries/"
gterm_index = dict()
for g in genres:
    filepath = base_filepath + g + corpus_d
    cdicts_flist = [files for path, dirs, files in os.walk(filepath)]
    cdicts_flist = cdicts_flist[0]
    corpus_dict = merge_to_global_dict(cdicts_flist, filepath, force_lower_case=lower_case)
    print("%s Dictionary has been loaded" % g )
    gterm_index = merge_global_dicts(gterm_index, corpus_dict)
    print("%s merged to Global Term Index" % g)
print( "Global Index Size: %s\n" % len(gterm_index))
#gterm_index = merge_global_dicts(corpus_dict, corpus_dict2) #, corpus_dict3, corpus_dict4)

pool = SynergeticPool( { '192.168.1.65':(40000,'123456'), '192.168.1.68':(40000,'123456') } )
print "Registering"
pool.register_mod(['experiments'])   
pool.register_mod(['trainevaloneclssvm'])
pool.register_mod(['vectorhandlingtools'])
print "Regitered OK"
exp = SVMExperiments()
vform = TermVectorFormat()

genres = [ "news" , "product_companies", "forum", "blogs", "wiki_pages"] 
base_filepath = "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/"
for g in genres:
    wpsl_genre = dict() 
    vectl_genre = dict()
    vform.glob_vect_l( [ g ] , gterm_index, None, wpsl_genre, vectl_genre, True)
    rest_genres = list()
    for rst_g in genres:
        if rst_g != g:
            rest_genres.append(rst_g)
    vform.glob_vect_l( rest_genres, gterm_index, 1000, wpsl_genre, vectl_genre, True)
    print len(wpsl_genre), len(vectl_genre)
    pool.dispatch(exp.set1, (base_filepath, g, vectl_genre, genres) )
    
    
print "Thank you and Goodbye!"

