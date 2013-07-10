

import sys
#sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
sys.path.append('../src')

import tables as tb

import html2vect.sparse.wngrams as h2v_wcng
import html2vect.sparse.cngrams as h2v_cng

from base.paramgridcrossval import ParamGridCrossValBase
from wrappedmodels.rfse import RFSE_Wrapped, cosine_similarity
import tables as tb
    
  
corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
#corpus_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/"

#kfolds_vocs_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/Kfolds_Vocabularies_4grams"
kfolds_vocs_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Kfolds_Vocs_Inds_4Grams"

genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
#genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop" ]

method_results = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None_TEST_NOBAGG.h5', 'w')
#method_results = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Char4Grams-Koppels-Bagging_method_kfolds-10_GridSearch_TEST.h5', 'w')

params_range = {
    'kfolds' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'vocab_size' : [100000], #[10000, 50000, 100000],
    'features_size' : [1000, 5000, 10000, 70000],
    'Iterations' : [100],
    'Sigma' : [0.5],
    #'Bagging' : [0.66],
} 

#word_n_gram_size = 1
#sparse_wng = h2v_wcng.Html2TF(word_n_gram_size, attrib='text', lowercase=True, valid_html=False)

char_n_gram_size = 4
sparse_cng = h2v_cng.Html2TF(char_n_gram_size, attrib='text', lowercase=True, valid_html=False)

ml_model = RFSE_Wrapped(cosine_similarity, -1.0, genres, bagging=False)

pgrid_corssv = ParamGridCrossValBase(\
                    ml_model, sparse_cng, method_results, 
                    genres, corpus_filepath, kfolds_vocs_filepath\
               )
               
html_file_l, cls_gnr_tgs = pgrid_corssv.corpus_files_and_tags()

results_h5 = pgrid_corssv.evaluate(html_file_l, cls_gnr_tgs, None, params_range, 'utf-8')

print results_h5

method_results.close()
