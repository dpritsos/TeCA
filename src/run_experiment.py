"""

"""
import os
from vectorhandlingtools import *
import sys
sys.path.append('../../synergeticprocessing/src')
from synergeticpool import *
from experiments import *

#pool = SynergeticPool( { '192.168.1.68':(40000,'123456'), '192.168.1.65':(40000,'123456') }, local_workers=1, syn_listener_port=41000 ) 
pool = SynergeticPool( local_workers=1, syn_listener_port=41000 )
print "Registering"
#pool.register_mod( ['experiments', 'vectorhandlingtools', 'trainevaloneclssvm'] )  
print "Regitered OK"
exp = SVMExperiments()

genres = [ "wiki_pages", "blogs", "news" , "product_companies", "forum" ] #academic
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
    exps_report.append( pool.dispatch(exp.tf_experiment_set1, base_filepath, corpus_d, vectors_d, g, genres, lower_case) )
    print("Experiment %s VS All: Dispatched" % g)

for exp_rep in exps_report:
    print exp_rep.value    

pool.join_all()

print "Thank you and Goodbye!"

