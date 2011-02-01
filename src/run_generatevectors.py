"""

"""
import os
from generatevectors import HTML2TF
import sys
sys.path.append('../../synergeticprocessing/src')
from synergeticpool import SynergeticPool

#pool = SynergeticPool( { '192.168.1.65':(40000,'123456'), '192.168.1.68':(40000,'123456') }, local_workers=1, syn_listener_port=41000 ) 
pool = SynergeticPool( local_workers=1, syn_listener_port=41000 )
print "Registering"
#pool.register_mod( ['html2vector', 'vectorhandlingtools', 'generatevectors'] )  
print "Regitered OK"

genres = [ "wiki_pages" ] #"news", "product_companies", "forum", "blogs", "wiki_pages" ] #academic ,  
#genres = [ "blogs" ] #academic "news"
base_filepath = ["/home/dimitrios/Documents/Synergy-Crawler/saved_pages", "../Documents/Synergy-Crawler/saved_pages"] 

html2tf = HTML2TF()

resaults = list()
for g in genres:
    filepath = str( "/" + g + "/html_pages/")
    tfv_file = str( "/" + g + "/corpus_webpage_vectors/" + g + ".tfvl" )
    tfd_file = str( "/" + g + "/corpus_dictionaries/" + g + ".tfd" )
    err_file = str( "/" + g + "/" + g + ".lst.err")
    #if filepath and not os.path.isdir((filepath + "corpus_dictionaries/")):
    #    os.mkdir((filepath + "corpus_dictionaries/"))
    #if filepath and not os.path.isdir((filepath + "corpus_webpage_vectors/")):
    #    os.mkdir((filepath + "corpus_webpage_vectors/"))
    #if filepath and not os.path.isdir((filepath + "ngrams_corpus_dictionaries/")):
    #    os.mkdir((filepath + "ngrams_corpus_dictionaries/"))
    #if filepath and not os.path.isdir((filepath + "ngrams_corpus_webpage_vectors/")):
    #    os.mkdir((filepath + "ngrams_corpus_webpage_vectors/"))
    #
    resaults.append( pool.dispatch(\
    html2tf.html2tfv_n_tfd, base_filepath, filepath, tfv_file, tfd_file, err_file, encoding='utf-8', error_handling='replace', low_mem='True' \
                     ) )
       
for res in resaults:
    print res.value             

pool.join_all()

print "Thank you and Goodbye!"

