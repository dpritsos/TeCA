"""

"""
import os
from vectorhandlingtools import *
import sys
sys.path.append('../../synergeticprocessing/src')
from synergeticpool import *
from experiments import *

pool = SynergeticPool( { '192.168.1.65':(40000,'123456'), '192.168.1.68':(40000,'123456') } ) 
print "Registering"
pool.register_mod( ['experiments', 'vectorhandlingtools', 'trainevaloneclssvm'] )  
print "Regitered OK"
exp = SVMExperiments()
vform = TermVectorFormat()

genres = [ "news" , "product_companies", "forum", "blogs", "wiki_pages" ] #academic
#genres_incmp = [ "blogs" ] #academic
base_filepath = ["/home/dimitrios/Documents/Synergy-Crawler/saved_pages/", "../Documents/Synergy-Crawler/saved_pages/"] 
corpus_d = "/corpus_dictionaries/"
vectors_d = "/corpus_webpage_vectors/"
##
ng_vectors_d = "/ngrams_corpus_webpage_vectors/"
ng_corpus_d = "/ngrams_corpus_dictionaries/"
lower_case = True

exps_report = list()
for g in genres:
    exps_report.append( pool.dispatch(exp.set1, base_filepath, ng_corpus_d, ng_vectors_d, g, genres, lower_case) )
    print("Experiment %s VS All: Dispatched" % g)

for exp_rep in exps_report:
    print exp_rep.value    

pool.join_all()

print "Thank you and Goodbye!"

