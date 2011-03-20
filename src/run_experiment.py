"""

"""
import os
from vectorhandlingtools import *
import sys
sys.path.append('../../synergeticprocessing/src')
from synergeticpool import *
from experiments import *

#pool = SynergeticPool( { '192.168.1.68':(40000,'123456') }, local_workers=1, syn_listener_port=41000 )
#pool = SynergeticPool( { '192.168.1.68':(40000,'123456'), '192.168.1.65':(40000,'123456') }, local_workers=1, syn_listener_port=41000 ) 
pool = SynergeticPool( local_workers=1, syn_listener_port=41000 )
print "Registering"
#pool.register_mod( [ 'basicfiletools', 'vectorhandlingtools', 'trainevaloneclssvm', 'experiments'] )  
print "Regitered OK"
exp = SVMExperiments()

genres = [ "wiki_pages", "news", "product_companies", "forum", "blogs"] #academic , "forum",   
#genres = [ "blog"] #, "eshop", "faq", "frontpage", "listing", "php", "spage"] 
#genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop"] 
base_filepath = ["/home/dimitrios/Documents/Synergy-Crawler/saved_pages/", "../Documents/Synergy-Crawler/saved_pages/"]
#base_filepath = ["/home/dimitrios/Documents/Synergy-Crawler/Santini_corpus/", "../Documents/Synergy-Crawler/Santini_corpus/"]
#base_filepath = ["/home/dimitrios/Documents/Synergy-Crawler/KI-04/", "../Documents/Synergy-Crawler/KI-04/"] 
train_tf_d = "/train_tf_dictionaries/"
train_tf_vectors = "/train_tf_vectors/"
test_tf_d = "/test_tf_dictionaries/"
test_tf_vectors = "/test_tf_vectors/"
##
train_nf_d = "/train_nf_dictionaries/"
train_nf_vectors = "/train_nf_vectors/"
test_nf_d = "/train_nf_dictionaries/"
test_nf_vectors = "/test_nf_vectors/"


exps_report = list()
#for g in genres:
#    exps_report.append( pool.dispatch(exp.tf_experiment_set1, base_filepath, train_tf_vectors, 140, test_tf_vectors, 49, g, genres, lower_case=True) )
#    print("Experiment %s VS All: Dispatched" % g)

#for g in genres:
#    exps_report.append( pool.dispatch(exp.tf_experiment_set3, (7000, 100),base_filepath, train_tf_vectors, 140, test_tf_vectors, 1000, g, genres, lower_case=True) )
#    print("Experiment %s VS All: Dispatched" % g)
    
for g in genres:
    exps_report.append( pool.dispatch(exp.tf_experiment_set4, base_filepath, train_tf_vectors, 2500, test_tf_vectors, 800, g, genres, freq_init=5, freq_lim=200, freq_step=5,lower_case=True, keep_terms=None) )
    print("Experiment %s VS All: Dispatched" % g)

#exps_report.append( pool.dispatch(exp.tf_experiment_set2, base_filepath, train_tf_vectors, 1000, test_tf_vectors, 500, genres, lower_case=True) )
#print("Experiment Multiclass SVM Dispatched")

for exp_rep in exps_report:
    print exp_rep.value    

pool.join_all()

print "Thank you and Goodbye!"

