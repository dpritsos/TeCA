"""

"""
import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
import numpy as np
import tables as tb
import html2tf.tables.cngrams as cng_tb
import html2tf.tables.tbtools as tbtls
from html2tf.dictionaries.tfdtools import TFDictHandler
import sklearn.svm as svm 
import scipy.sparse as sps
from trainevaloneclssvm import SVMTE 


class ResaultsTable_desc(tb.IsDescription):
    kfold = tb.UInt32Col(pos=1)
    nu = tb.Float32Col(pos=2)
    feat_num = tb.UInt32Col(pos=3)
    F1 = tb.Float32Col(pos=4)
    P = tb.Float32Col(pos=5)
    R = tb.Float32Col(pos=6)
    Acc = tb.Float32Col(pos=7)
   

class CSVM_CrossVal(tbtls.TFTablesHandler):
    
    def __init__(self, h5file, corpus_grp, trms_type_grp):
        tbtls.TFTablesHandler.__init__(self, h5file)
        self.h5file = h5file
        self.corpus_grp = corpus_grp
        self.trms_type_grp = trms_type_grp
        self.genres_lst = list()
        self.kfold_chnk = dict()
        self.page_lst_tb = dict()
        self.kfold_mod = dict()
        self.gnr2clss = dict()
    
    def complementof_list(self, lst, excld_dwn_lim, excld_up_lim):
        if excld_dwn_lim == 0:
            return lst[excld_up_lim:]
        if excld_up_lim == len(lst):
            return lst[0:excld_dwn_lim]
        inv_lst = np.concatenate((lst[0:excld_dwn_lim], lst[excld_up_lim:]))
        return inv_lst
            
    def data_preparation(self, h5f2save, kfolds, format):
        corpus = self.h5file.getNode('/', self.corpus_grp)
        self.genres_lst = corpus._v_attrs.genres_lst
        for i, gnr in enumerate(self.genres_lst):
            self.page_lst_tb[gnr] = self.h5file.getNode( self.corpus_grp + self.trms_type_grp + gnr, '/PageListTable' )
            self.kfold_chnk[gnr] = np.divide(len(self.page_lst_tb[gnr]),kfolds)
            self.kfold_mod[gnr] = np.divide(len(self.page_lst_tb[gnr]),kfolds)
            self.gnr2clss[gnr] = i + 1
        
        for k in range(kfolds):
            start = k*self.kfold_chnk[gnr]
            end = (k+1)*self.kfold_chnk[gnr]
            if (k+1) == kfolds:
                end = end + self.kfold_mod[gnr]
            
            training_pg_lst = list()
            training_clss_tag_lst = list()
            crossval_pg_lst = list()
            crossval_clss_tag_lst = list()
            
            for gnr in self.genres_lst:
                page_lst_tb = self.page_lst_tb[gnr].read()
                #Get the Evaluation set for this Genre and later concatenate it to the rest of the Evaluation Genres
                grn_page_lst_tb = page_lst_tb['table_name'][start:end]
                crossval_pg_lst.extend( page_lst_tb['table_name'][start:end] )
                crossval_clss_tag_lst.extend( [self.gnr2clss[gnr]]*len(grn_page_lst_tb) )
                #Get the Training set for this Genre g
                grn_trn_pg_lst = self.complementof_list( page_lst_tb['table_name'], start, end )
                training_pg_lst.extend( grn_trn_pg_lst )
                training_clss_tag_lst.extend( [self.gnr2clss[gnr]]*len(grn_trn_pg_lst) )
            
            
            
            print len(crossval_pg_lst)
            print len(crossval_clss_tag_lst), crossval_clss_tag_lst 
            print len(training_pg_lst), 
            print len(training_clss_tag_lst), training_clss_tag_lst 
            
            
            
            
            #Create the Training-set Dictionary - Sorted by frequency
            print "Creating Dictionary - Sorted by Frequency" 
            term_idx_d, freq_arr = self.TFTables2TFDict_n_TFArr(self.corpus_grp + self.trms_type_grp,\
                                                                training_pg_lst,\
                                                                data_type=tbtls.default_TF_3grams_dtype)
            
            print term_idx_d.items()[0:10]
            print freq_arr[0:10] 
            0/0
            
            """
            idx_freq_lst = list(idx_freq_d.items())
            idx_freq_lst.sort()
            DicFreq = h5f_exp_dat.createArray( h5f_exp_dat.root, 'DictFreq_'+str(k), np.array([idxf_pair[1] for idxf_pair in idx_freq_lst]) )
            #Create the Training-Set Array
            print "Preparing Training Set", g
            filters = tb.Filters(complevel=5, complib='zlib')
            training_EArr = h5f_exp_dat.createEArray(h5f_exp_dat.root, 'Training_'+str(k), tb.Float32Atom(), (0,len(term_idx_d)), filters=filters)
            training_EArr = self.tb_hdlr.pagetf_EArray(h5f_exp_dat, training_EArr, fileh, base_tbgroup + g, training_lst, term_idx_d, data_type=np.float32)
            #training_arr = self.tb_hdlr.pagetf_array(fileh, base_tbgroup + g, training_lst, term_idx_d, data_type=np.float32
            #Save Training Data in CArray (HD5 PyTables file)
            print training_EArr[0:1, 10:2]
            print np.shape(training_EArr)
            #Create the Evaluaton-Set Array
            print "Prepare Eval Set", g
            k_eval_EArr = h5f_exp_dat.createEArray(h5f_exp_dat.root, 'Eval_0_'+str(k), tb.Float32Atom(), (0,len(term_idx_d)), filters=filters)
            k_eval_EArr = self.tb_hdlr.pagetf_EArray(h5f_exp_dat.root, k_eval_EArr, fileh, base_tbgroup + g, k_eval_set_lst, term_idx_d, data_type=np.float32)
            k_eval_EArr = h5f_exp_dat.createEArray(h5f_exp_dat.root, 'Eval_1_'+str(k), tb.Float32Atom(), (0,len(term_idx_d)), filters=filters)            
            for gnr in genres:
                if gnr != g:
                    print "Prepare Eval Set", gnr
                    PgLstTbl = fileh.getNode( base_tbgroup + g, '/PageListTable' )
                    PgLstArr = PgLstTbl.read()
                    k_eval_EArr = self.tb_hdlr.pagetf_EArray(h5f_exp_dat, k_eval_EArr, fileh, base_tbgroup + g, PgLstArr['table_name'], term_idx_d, data_type=np.float32)
            del PgLstArr       
            print training_EArr[0:1, 10:20], np.shape(training_EArr)
            print k_eval_EArr[0:1, 10:20], np.shape(k_eval_EArr)
            """
            #start = end
            #end = end + fold_size   
             
            
            
    def exec_test(self):
        self.data_preparation(None, 10, None)



class SVMExperiments_with_tables(object):
    
    def __init__(self):
        self.keep_atleast_history = 0
        #self.tb_hdlr = TFTablesHandler()
        #self.tfdhdlr = TFdictHandler()
    
    def complementof_list(self, lst, excld_dwn_lim, excld_up_lim):
        if excld_dwn_lim == 0:
            return lst[excld_up_lim:]
        if excld_up_lim == len(lst):
            return lst[0:excld_dwn_lim]
        inv_lst = np.concatenate((lst[0:excld_dwn_lim], lst[excld_up_lim:]))
        return inv_lst
    
    def kFold_Cross_data_preparation(self, h5_exp_data_path, kfolds, nu_lst, featr_size_lst, fileh, base_tbgroup, g, genres):
        #Create a Experiment's Group and CArrays for high scalable test
        h5f_exp_dat = tb.openFile( h5_exp_data_path+'_'+g+'.h5', 'w') 
        #Load the Page-List Table for Genre g in order to select Page-Tables to be loaded as Training/Evaluation set for each fold 
        PageListTable = fileh.getNode( base_tbgroup + g, '/PageListTable' )
        PageListArray = PageListTable.read()
        #Calculate the fold Size
        fold_size = int( np.math.floor( len(PageListArray) / kfolds ) )# it should be Integer
        start = 0
        end = fold_size
        for k in range(kfolds):
            #Get the Evaluation set for this Genre and later concatenate it to the rest of the Evaluation Genres
            k_eval_set_lst = PageListArray['table_name'][start:end]
            #Get the Training set for this Genre g
            training_lst = self.complementof_list( PageListArray['table_name'], start, end )
            #Create the Training-set Dictionary - Sorted by frequency
            print "Creating Dictionary - Sorted by Frequency" 
            term_idx_d, idx_freq_d = self.tb_hdlr.TFtbls_Lst_2_TIdx_D(fileh, base_tbgroup + g, training_lst, data_type=tbtls.default_TF_3grams_dtype)
            idx_freq_lst = list(idx_freq_d.items())
            idx_freq_lst.sort()
            DicFreq = h5f_exp_dat.createArray( h5f_exp_dat.root, 'DictFreq_'+str(k), np.array([idxf_pair[1] for idxf_pair in idx_freq_lst]) )
            #Create the Training-Set Array
            print "Preparing Training Set", g
            filters = tb.Filters(complevel=5, complib='zlib')
            training_EArr = h5f_exp_dat.createEArray(h5f_exp_dat.root, 'Training_'+str(k), tb.Float32Atom(), (0,len(term_idx_d)), filters=filters)
            training_EArr = self.tb_hdlr.pagetf_EArray(h5f_exp_dat, training_EArr, fileh, base_tbgroup + g, training_lst, term_idx_d, data_type=np.float32)
            #training_arr = self.tb_hdlr.pagetf_array(fileh, base_tbgroup + g, training_lst, term_idx_d, data_type=np.float32
            #Save Training Data in CArray (HD5 PyTables file)
            print training_EArr[0:1, 10:2]
            print np.shape(training_EArr)
            #Create the Evaluaton-Set Array
            print "Prepare Eval Set", g
            k_eval_EArr = h5f_exp_dat.createEArray(h5f_exp_dat.root, 'Eval_0_'+str(k), tb.Float32Atom(), (0,len(term_idx_d)), filters=filters)
            k_eval_EArr = self.tb_hdlr.pagetf_EArray(h5f_exp_dat.root, k_eval_EArr, fileh, base_tbgroup + g, k_eval_set_lst, term_idx_d, data_type=np.float32)
            eval_EArr = h5f_exp_dat.createEArray(h5f_exp_dat.root, 'Eval_1_'+str(k), tb.Float32Atom(), (0,len(term_idx_d)), filters=filters)            
            for gnr in genres:
                if gnr != g:
                    print "Prepare Eval Set", gnr
                    PgLstTbl = fileh.getNode( base_tbgroup + g, '/PageListTable' )
                    PgLstArr = PgLstTbl.read()
                    eval_EArr = self.tb_hdlr.pagetf_EArray(h5f_exp_dat, eval_EArr, fileh, base_tbgroup + g, PgLstArr['table_name'], term_idx_d, data_type=np.float32)
            del PgLstArr       
            print training_EArr[0:1, 10:20], np.shape(training_EArr)
            print k_eval_EArr[0:1, 10:20], np.shape(k_eval_EArr)
            start = end
            end = end + fold_size   
        h5f_exp_dat.close()       
        
    def OCSVM_kFold_Cross(self, kfolds, nu_lst, featr_size_lst, fileh, base_tbgroup, res_table, g, genres):
        ##
        for k in range(kfolds):
            #Load the Training-set Dictionary - Sorted by frequency for THIS-FOLD
            print "Get Dictionary - Sorted by Frequency for this fold"
            DicFreq = fileh.getNode(fileh.root, '/DictFreq_'+str(k))
            #Get Training-set for This-Fold
            print "Get Training Data Array"
            training_EArr = fileh.getNode(fileh.root, 'Training_'+str(k))
            #Get the Evaluation-set for This-FOLD
            print "Get Evaluation Data Array"
            k_eval_EArr = fileh.getNode(fileh.root, 'Eval_0_'+str(k))
            print "Get Evaluation Data Array"
            eval_EArr = fileh.getNode(fileh.root, 'Eval_1_'+str(k))
            for featrs_size in featr_size_lst: 
                #Keep the amount of feature required - it will keep_at_least as many as
                #the featrs_size keeping all the terms with same frequency the last term satisfies the featrs_size
                feat_len = np.max(np.where( DicFreq == DicFreq[featrs_size] )[0])
                print "Features Size:", feat_len
                for nu in nu_lst:
                    ocsvm = svm.OneClassSVM(kernel='rbf', nu=nu)
                    print "FIT model"
                    feat_len =  len(training_EArr[1,:])
                    print len(k_eval_EArr[:, 0:feat_len])
                    arrr = np.where( k_eval_EArr[:, 0:feat_len] > 2, k_eval_EArr[:, 0:feat_len], 0)
                    arrr[ np.nonzero(arrr) ] = 1 
                    print arrr 
                    ocsvm.fit( arrr ) #class_weight={}, sample_weight=None, **params)
                    print "Predict for kfold"
                    arrr = np.where( training_EArr[:, 0:feat_len] > 2, training_EArr[:, 0:feat_len], 0)
                    arrr[ np.nonzero(arrr) ] =  1
                    res = ocsvm.predict( arrr )
                    tp = len( np.where( res == 1 )[0] )
                    fn = len( np.where( res == -1 )[0] )
                    print res, tp, fn      
                    print "Predict for rest of genres"
                    arrr = np.where( eval_EArr[:, 0:feat_len] > 2, eval_EArr[:, 0:feat_len], 0)
                    arrr[ np.nonzero(arrr) ] =  1
                    res = ocsvm.predict( arrr )
                    #res = ocsvm.predict( sps.csr_matrix( (eval_EArr[np.nonzero(eval_EArr[:, 0:feat_len])], np.nonzero(eval_EArr[:, 0:feat_len])) ) )
                    print "Evaluate"
                    tn = len( np.where( res == -1 )[0] )
                    ###print np.where( res == 1 )[0] 
                    fp = len( np.where( res == 1 )[0] )
                    print res, tn, fp
                    
                    if (tp + fp) != 0.0:
                        P = np.float(tp) / np.float( tp + fp )
                    else:
                        P = 0.0
                    if (tp + fn) != 0.0:
                        R = np.float(tp) / np.float( tp + fn )
                    else:
                        R = 0.0
                    if (P + R) != 0.0:
                        F1 = np.float( 2 * P * R ) / np.float( P + R )
                    else:
                        F1 = 0.0
                    if (tp + tn + fp + fn) !=0:
                        Acc = np.float(tp + tn) / np.float(tp + tn + fp + fn)
                    else:
                        Acc = 0.0
                    print tn, fn, tn, fp, F1, P, R, Acc
                    res_table.row['kfold'] = k
                    res_table.row['nu'] = nu
                    res_table.row['feat_num'] = feat_len
                    res_table.row['F1'] = F1
                    res_table.row['P'] = P
                    res_table.row['R'] = R
                    res_table.row['Acc'] = Acc
                    res_table.row.append()
        res_table.flush()
        
        
    def CSVM_kFold_Cross(self, kfolds, C, featr_size_lst, fileh, base_tbgroup, res_table, g, genres):
        ##
        for k in range(kfolds):
            #Load the Training-set Dictionary - Sorted by frequency for THIS-FOLD
            print "Get Dictionary - Sorted by Frequency for this fold"
            DicFreq = fileh.getNode(fileh.root, '/DictFreq_'+str(k))
            #Get Training-set for This-Fold
            print "Get Training Data Array"
            training_EArr = fileh.getNode(fileh.root, 'Training_'+str(k))
            #Get the Evaluation-set for This-FOLD
            print "Get Evaluation Data Array"
            k_eval_EArr = fileh.getNode(fileh.root, 'Eval_0_'+str(k))
            print "Get Evaluation Data Array"
            eval_EArr = fileh.getNode(fileh.root, 'Eval_1_'+str(k))
            for featrs_size in featr_size_lst: 
                #Keep the amount of feature required - it will keep_at_least as many as
                #the featrs_size keeping all the terms with same frequency the last term satisfies the featrs_size
                feat_len = np.max(np.where( DicFreq == DicFreq[featrs_size] )[0])
                print "Features Size:", feat_len
                for nu in nu_lst:
                    ocsvm = svm.SVC()
                    print "FIT model"
                    feat_len =  len(training_EArr[1,:])
                    print len(k_eval_EArr[:, 0:feat_len])
                    arrr = np.where( k_eval_EArr[:, 0:feat_len] > 2, k_eval_EArr[:, 0:feat_len], 0)
                    arrr[ np.nonzero(arrr) ] = 1
                    lbls = np.ones(len(arrr))
                    
                    arrr1 = np.where( eval_EArr[:500, 0:feat_len] > 2, eval_EArr[:500, 0:feat_len], 0)
                    arrr1[ np.nonzero(arrr) ] =  1
                    lbls1 = np.ones(len(arrr1))
                    lbls1[np.where(lbls1 > 0)] = 2
                     
                    print np.shape(np.vstack((arrr, arrr1)))
                    print np.shape(np.hstack((lbls, lbls1)))
                    ocsvm.fit( np.vstack((arrr, arrr1)), np.hstack((lbls, lbls1)) ) #class_weight={}, sample_weight=None, **params)
                    
                    print "Predict for kfold"
                    arrr = np.where( training_EArr[:, 0:feat_len] > 2, training_EArr[:, 0:feat_len], 0)
                    arrr[ np.nonzero(arrr) ] =  1
                    res = ocsvm.predict( arrr )
                    tp = len( np.where( res == 1 )[0] )
                    fn = len( np.where( res == 2 )[0] )
                    print res, tp, fn      
                    print "Predict for rest of genres"
                    arrr = np.where( eval_EArr[500:, 0:feat_len] > 2, eval_EArr[500:, 0:feat_len], 0)
                    arrr[ np.nonzero(arrr) ] =  1
                    res = ocsvm.predict( arrr )
                    
                    print "Evaluate"
                    tn = len( np.where( res == 2 )[0] )
                    ###print np.where( res == 1 )[0] 
                    fp = len( np.where( res == 1 )[0] )
                    print res, tn, fp
                    
                    if (tp + fp) != 0.0:
                        P = np.float(tp) / np.float( tp + fp )
                    else:
                        P = 0.0
                    if (tp + fn) != 0.0:
                        R = np.float(tp) / np.float( tp + fn )
                    else:
                        R = 0.0
                    if (P + R) != 0.0:
                        F1 = np.float( 2 * P * R ) / np.float( P + R )
                    else:
                        F1 = 0.0
                    if (tp + tn + fp + fn) !=0:
                        Acc = np.float(tp + tn) / np.float(tp + tn + fp + fn)
                    else:
                        Acc = 0.0
                    print tn, fn, tn, fp, F1, P, R, Acc
                    res_table.row['kfold'] = k
                    res_table.row['nu'] = nu
                    res_table.row['feat_num'] = feat_len
                    res_table.row['F1'] = F1
                    res_table.row['P'] = P
                    res_table.row['R'] = R
                    res_table.row['Acc'] = Acc
                    res_table.row.append()
        res_table.flush()
        
        

if __name__=='__main__':
    

 
    exp = SVMExperiments_with_tables()
    kfolds = 10
    nu_lst = [0.2, 0.8]
    featr_size_lst = [1000]
    fileh = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Santini_corpus.h5', 'r')
    genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
    
    csvm_crossval = CSVM_CrossVal(fileh, "/Santini_corpus", "/trigrams/")
    csvm_crossval.exec_test()
    
    fileh.close() 
    
"""    
    #fileh = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/ACC.h5', 'r')
    #fileh = tb.openFile('/home/dimitrios/Synergy-Crawler/Santini_corpus/Santini_corpus.h5', 'r')
    #res_table = fileh.getNode('/Santini_corpus', 'Res_10FldC_3grams_Freq_OCSVM') 
    #res_table =  fileh.getNode('/Automated_Crawled_Corpus', 'Res_10FldC_3grams_Freq_OCSVM')
    res_fileh = tb.openFile('/home/dimitrios/Synergy-Crawler/Santini_corpus/Res_10FldC_3grams_Freq_OCSVM.h5', 'w')
    #res_fileh = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/Res_10FldC_3grams_Freq_OCSVM.h5', 'w')
    #base_tbgroup = "/Automated_Crawled_Corpus/trigrams/"
    base_tbgroup = "/Santini_corpus/trigrams/"
    #genres = [ "blogs", "forum", "news", "product_pages", "wiki_pages" ] 
    
    for g in genres:
        #exp.kFold_Cross_data_preparation('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/10fold_Exp_Genre_' , kfolds, nu_lst, featr_size_lst, fileh, base_tbgroup, g, genres)
        #exp.kFold_Cross_data_preparation('/home/dimitrios/Synergy-Crawler/Santini_corpus/10fold_Exp_Genre_' , kfolds, nu_lst, featr_size_lst, fileh, base_tbgroup, g, genres)
        h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Santini_corpus/10fold_Exp_Genre__'+g+'.h5', 'r')
        #h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/10fold_Exp_Genre__'+g+'.h5', 'r')
        res_table =  res_fileh.createTable('/', 'Res_'+g+'_vs_rest', ResaultsTable_desc)
        #exp.CSVM_kFold_Cross(kfolds, 1, featr_size_lst, h5file, base_tbgroup, res_table, g, genres)
        exp.OCSVM_kFold_Cross(kfolds, nu_lst, featr_size_lst, h5file, base_tbgroup, res_table, g, genres)
        h5file.close()
    res_fileh.close()
    #fileh.close()

"""
    
    
    
    
    
    
    
    
    
    
    

