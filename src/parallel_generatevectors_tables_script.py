""" """

import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
from generatevectors_tables import Html2TF_Concurrent
import html2tf.tables.tbtools as tbtls
#from synergeticpool import SynergeticPool

#pool = SynergeticPool( { '192.168.1.65':(40000,'123456'), '192.168.1.68':(40000,'123456') }, local_workers=1, syn_listener_port=41000 ) 
#pool = SynergeticPool( local_workers=1, syn_listener_port=41000 )
#print "Registering"
#pool.register_mod( ['html2vector', 'vectorhandlingtools', 'generatevectors'] )  
#print "Regitered OK"

#genres = [ "blogs", "forum", "news", "product_pages", "wiki_pages" ] #academic , "news", "wiki_pages", "product_companies", "blogs", "forum"  
genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
#genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop"] 
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus", "../Synergy-Crawler/Automated_Crawled_Corpus"]
base_filepath = ["/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre", "../Synergy-Crawler/Santinis_7-web_genre"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Santini_corpus_html2txt"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/KI-04", "../Synergy-Crawler/KI-04"]  

##base_filepath = ["/home/dimitrios/Synergy-Crawler/Santini_corpus_html2txt/"]
##genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
#genres = [ "blog_pgs", "news_pgs", "product_pgs", "forum_pgs", "wiki_pgs" ] 
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Crawled_corpus_3000/"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Manually_Selected_Crawled_corpus_75/"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Crawled_corpus_500/"]


html2tf = Html2TF_Concurrent( lowercase=True, valid_html=False )
html2nf = Html2TF_Concurrent(n=3, lowercase=True, valid_html=False)

#if filepath and not os.path.isdir((filepath + "corpus_dictionaries/")):
#    os.mkdir((filepath + "corpus_dictionaries/"))
          
resaults = list()

#Creating the Default H5File tree structure to save the Corpus' TF and other Tables(pytables)
CorpusTable = tbtls.TFTablesHandler()   
#CorpusTable.create(table_name="/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/ACC.h5",\
#ttypes_structures_lst=["trigrams"], corpus_name="Automated_Crawled_Corpus", genres_lst=genres)
CorpusTable.create(table_name="/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Santini_corpus.h5",\
                   ttypes_structures_lst=["trigrams"], corpus_name="Santini_corpus", genres_lst=genres)

print CorpusTable.get()

for g in genres:
    #Vectors file paths
    #filepath = str( "/" + g + "/html2ascii_perl_text/")
    #filepath = str( "/" + g + "/html2text_debian_text/")
    #filepath = str( "/" + g + "/htmldetagger_console_ver_text/")
    #filepath = str( "/" + g + "/htmldetagger_console_500_ver_text/")
    #filepath = str( "/" + g + "/txt_rapidminer_app/")
    #filepath = str( "/" + g + "/txt_Htmlremover_app/")
    #filepath = str( "/" + g + "/txt_html2vectors_mod/")
    #filepath = str( "/" + g + "/txt_html2vectors_mod_500/")
    #filepath = str( "/" + g + "/nltk-clean_html_text/")
    #filepath = str( "/" + g + "/lxml_elementtree_text/")
    #filepath = str( "/" + g + "/lxml_elementtree_text_500/")
    filepath = str( "/" + g + "/html/")
    #tfv_file = str( "/" + g + "/html2ascii_perl_ng-tfv/" + g + ".nfvl" )

    #resaults.append( pool.dispatch(\    
    print html2nf.exec_for(base_filepath, filepath, CorpusTable.get(), "/Santini_corpus/trigrams/"+g, "PageListTable", load_encoding='utf-8', error_handling='replace')
    #print html2nf.exec_for(base_filepath, filepath, CorpusTable.get(), "/Automated_Crawled_Corpus/trigrams/"+g, "PageListTable", load_encoding='utf-8', error_handling='replace')
    #                 ) )
    #resaults.append( pool.dispatch(\
    #print html2nf.exec_for(base_filepath, test_filepath, test_tfv_file, test_tfd_file, test_err_file, load_encoding='utf-8', save_encoding='utf-8', error_handling='replace', low_mem='True')
    #                 ) )
       
#for res in resaults:
#    print res.value

CorpusTable.get().close()


#pool.join_all()

print "Thank you and Goodbye!"


