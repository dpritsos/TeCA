"""

"""
import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
import numpy as np
import tables as tb
import html2tf.tables.cngrams as cng_tb
import html2tf.tables.tbtools as tbtls
from html2tf.dictionaries.tfdtools import TFdictHandler
import scikits.learn.svm as svm 



class ResaultsTable_desc(tb.IsDescription):
    nu = tb.Float32Col(pos=1)
    feat_num = tb.UInt64Col(pos=2)
    F1 = tb.Float32Col(pos=3)
    P = tb.Float32Col(pos=4)
    R = tb.Float32Col(pos=5)
   

class SVMExperiments_with_tables(object):
    
    def __init__(self):
        self.svm = svm.OneClassSVM()
        self.keep_atleast_history = 0
        self.tb_hdlr = tbtls.TFTablesHandler()
        self.tfdhdlr = TFdictHandler()
    
    def complementof_list(self, lst, excld_dwn_lim, excld_up_lim):
        if excld_dwn_lim == 0:
            return lst[excld_up_lim:]
        if excld_up_lim == len(lst):
            return lst[0:excld_dwn_lim]
        inv_lst = lst[0:excld_dwn_lim] + lst[excld_up_lim:]
        return inv_lst
        
    def OCSVM_kFold_Cross(self, kfolds, nu_lst, featr_size_lst, fileh, base_tbgroup, res_table, g, genres):
        #Create a Experiment's Group and CArrays for high scalable test
        h5f_tmp = tb.openFile('/home/dimitrios/Synergy-Crawler/Exp_tmp.h5', 'a') 
        #Load the Page-List Table for Genre g in order to select Page-Tables to be loaded as Training/Evaluation set for each fold 
        PageListTable = fileh.getNode( base_tbgroup + g, '/PageListTable' )
        PageListArray = PageListTable.read()
        #Calculate the fold Size
        fold_size = int( np.math.floor( len(PageListArray) / kfold ) )# it should be Integer
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
            #Create the Training-Set Array
            print "Preparing Training Set", g
            training_arr = self.tb_hdlr.pagetf_array(fileh, base_tbgroup + g, training_lst, term_idx_d, data_type=np.float32)
            #Save Training Data in CArray (HD5 PyTables file)
            filters = tb.Filters(complevel=5, complib='zlib')
            training_CArr = h5f_tmp.createCArray(h5f_tmp.root, 'Training_'+str(k), tb.Float32Atom(), training_arr.shape, filters=filters)
            training_CArr[:,:] = training_arr[:,:]
            print training_CArr[0:1, 10:20]
            print training_arr[0:1, 10:20]
            h5f_tmp.flush()
            print 'Flush'
            del training_arr
            print training_CArr[0:1, 10:20]
            print np.shape(training_CArr)
            #Create the Evaluaton-Set Array
            eval_arr = self.tb_hdlr.pagetf_array(fileh, base_tbgroup + g, k_eval_set_lst, term_idx_d, data_type=np.float32)            
            k_eval_EArr = h5f_tmp.createEArray(h5f_tmp.root, 'Eval_'+str(k), tb.Float32Atom(), (0, eval_arr.shape[1]), filters=filters)
            k_eval_EArr.append(eval_arr)
            h5f_tmp.flush()
            for gnr in genres:
                if gnr != g:
                    print "Prepare Training Set", gnr
                    PgLstTbl = fileh.getNode( base_tbgroup + g, '/PageListTable' )
                    PgLstArr = PgLstTbl.read() 
                    eval_arr = self.tb_hdlr.pagetf_array(fileh, base_tbgroup + g, PgLstArr['table_name'], term_idx_d, data_type=np.float32)
                    k_eval_EArr.append(eval_arr)
                    h5f_tmp.flush()
            #h5f_tmp.flush()
            #del eval_arr
            print training_CArr[0:1, 10:20], np.shape(training_CArr)
            print k_eval_EArr[0:1, 10:20], np.shape(k_eval_EArr)
            
            
            0/0
            for featrs_size in featr_size_lst: 
                #Keep the amount of feature required - it will keep_at_least as many as
                #the featrs_size keeping all the terms with same frequency the last term satisfies the featrs_size
                ###### ----> idx_freq_d <--- FOR KEEP AT LEAST
                tf_d = self.tfdhdlr.keep_atleast( dict(list( train_tf_d_arr )), featrs_size) #Change this function with a Numpy.array ready one
                #tf_d = dict(list( train_tf_d_arr ))
                #Convert TF Dictionary to Index Dictionary 
                
                
                    #for nu in nu_lst:
                    #    ocsvm = svm.OneClassSVM(kernel='rbf', nu=0.5)
                    #    ocsvm.fit(train_arr) #class_weight={}, sample_weight=None, **params)
                    #    res = ocsvm.predict(eval_arr)
                    #    tp = np.sum( np.where( res[0:k_eval_set_lst] == 1 ) ) 
                    #    fp = np.sum( np.where( res[0:k_eval_set_lst] == -1 ) )
                    #    P = np.float(tp) / np.float( tp + fp )
                    #    R = np.float(tp) / np.float( tp + fn )
                    #    F1 = np.float( 2 * P * R ) / np.float( P + R )
        
            start = end
            end = end + fold_size   
        h5f_tmp.close()
                
"""       
    def svm_evalon_tset(self, fobj, tf_dict, train_wpg_l, train_tf_d_l, test_wpg_l, test_tf_d_l, g, genres, format_l, nu_lst, freq_th=1, keep_terms=None):
        #If there is a limit to the terms of the dictionary that has to be kept
        if keep_terms:
            tf_dict = VHTools.keep_atleast(tf_dict, keep_terms)
            if self.keep_atleast_history == len(tf_dict):
                self.keep_atleast_history = len(tf_dict)
                return
            self.keep_atleast_history = len(tf_dict)
            fobj.write("^^^^ Terms kept= " + str(len(tf_dict)) + " ^^^^\n")
            fobj.write("TF_dictionary= " + str(tf_dict) + "\n")
        print("TF Dictionary kept length: %s" % len(tf_dict))
        #Convert TF Dictionary to Index Dictionary
        print("Convert TF Dictionary to Index Dictionary") 
        tidx_d = VHTools.tf2tidx(tf_dict)
        print('Training IdxF Dictionary size =  %s' % len(tidx_d))
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Training TF Vectors to IdxF Vectors")
        train_tf_d_l = VHTools.tf2idxf(train_tf_d_l, tidx_d)
        train_idxf_d_l = train_tf_d_l
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Test TF Vectors to IdxF Vectors")
        test_tf_d_l = VHTools.tf2idxf(test_tf_d_l, tidx_d)
        test_idxf_d_l = test_tf_d_l
        #Building a Index Frequency Dictionary 
        print("Building a Index Frequency Dictionary for this Genre: %s" % g) 
        idxf_d = VHTools.tf2idxf(tf_dict, tidx_d)
        #Start the experiments
        for i in format_l:
            if i == 1:
                fobj.write("**** Inverse Binary ****\n")
                train_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=freq_th)
                test_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(test_idxf_d_l), idxf_d, tf_threshold=freq_th)
            elif i == 2:
                fobj.write("**** Binary ****\n")
                train_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=freq_th)
                test_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(test_idxf_d_l), idxf_d, tf_threshold=freq_th)
            elif i == 3:
                fobj.write("**** Normalised by Max Term ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=True)
                test_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(test_idxf_d_l), div_by_max=True)
            elif i == 4:
                fobj.write("**** Normalised by Total Sum ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=False)             
                test_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(test_idxf_d_l), div_by_max=False)
            elif i == 5:
                fobj.write("**** Training Corpus Dictionary Frequency in TF Vectors ****\n")
                train_idxf_d_l_frmd = vformat.tf2tf_idf(copy.deepcopy(train_idxf_d_l), idxf_d)
                test_idxf_d_l_frmd = vformat.tf2tf_idf(copy.deepcopy(test_idxf_d_l), idxf_d)
            for nu in nu_lst: #[0.2, 0.3, 0.5, 0.7, 0.8]: # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                #Train SVM 
                fobj.write("++++ for nu= " + str(nu) + " ++++\n")
                print "Training"
                #for size in range(1500,9000,500):
                svm_m = SVMTE.train_oneclass_svm(fobj, train_idxf_d_l_frmd[0:], nu )
                eval_res_d = SVMTE.evaluate_oneclass_svm(svm_m, test_idxf_d_l_frmd, folds=len(genres), fld_seq_num=genres.index(g)) #len(genres))
                s = "+ %s, - %s\n" % (eval_res_d['c1'], eval_res_d['c2'])
                s += "tp=%s, tn=%s\nfp=%s, fn=%s\n" % (eval_res_d['tp'],eval_res_d['tn'],eval_res_d['fp'],eval_res_d['fn'])
                s += "Precision=%f\n" % eval_res_d['precision'] 
                s += "Recall=%f\n" % eval_res_d['recall'] 
                s += "F1=%s\n\n" % eval_res_d['f1']
                fobj.write(s)
                SVMTE.negative_indices(fobj, eval_res_d['p_labels'], len(genres))
    
    
    
    def load_tt_set(self, filepath, tt_path, tt_pg_lim, g, lower_case):
        #Load Training Vectors
        print("Load Training Vectors for: %s" % g)
        print filepath + tt_path
        tt_wpg_l, tt_tf_d_l = VHTools.load_tfd_l_frmpaths(None, [filepath + tt_path], page_lim=tt_pg_lim, force_lower_case=lower_case)
        ## Maybe the following shorting patch should be transfered to the Save function in vectohandlingtools module
        mrgd_l = list()
        for i in range(len(tt_wpg_l)):
            mrgd_l.append( (tt_wpg_l[i], tt_tf_d_l[i]) )
        ###print mrgd_l[0]
        mrgd_l.sort()
        ###print mrgd_l[0]
        tt_wpg_l = [g_wpg[0] for g_wpg in mrgd_l]
        tt_tf_d_l = [g_tf_d[1] for g_tf_d in mrgd_l]
        del mrgd_l 
        ## END of patch
        #Build Dictionary From Training Vectors
        print("Building TF Dictionary From Training Vectors")
        tf_dict = VHTools.gen_tfd_frmlist(tt_tf_d_l)
        return  { 'tt-set' : [tt_wpg_l, tt_tf_d_l], 'tf-dict': tf_dict } 
    
    def load_tt_sets(self, base, tt_path, tt_pg_lim, genres, lower_case):
        #Load Test Vectors
        print("Load Test Vectors")
        tt_wpg_l = list()
        tt_tf_d_l = list()
        for rst_g in genres:
            wpgl_tmp, tfdl_tmp = VHTools.load_tfd_l_frmpaths(None, [ base + rst_g + tt_path ], page_lim=0, force_lower_case=lower_case) #base + rst_g + test_path
            wpgl = wpgl_tmp[-tt_pg_lim:]
            tfdl = tfdl_tmp[-tt_pg_lim:]
            ## Maybe the following shorting patch should be transfered to the Save function in vectohandlingtools module
            mrgd_l = list()
            for i in range(len(wpgl)):
                mrgd_l.append( (wpgl[i], tfdl[i]) )
            mrgd_l.sort()
            wpgl = [wpgl[0] for wpgl in mrgd_l]
            tfdl = [tfdl[1] for tfdl in mrgd_l]
            del mrgd_l 
            ## END of patch
            tt_wpg_l.extend( wpgl ) 
            tt_tf_d_l.extend( tfdl )
        del wpgl, tfdl
        print('Test TF Vectors list size =  %s (%s)' % (len(tt_wpg_l), len(tt_tf_d_l)))
        return [tt_wpg_l, tt_tf_d_l]
    
    def exprmt_feature_len_variation(self, keep_term_lst, base_filepath, training_path, train_pg_lim, test_path, test_pg_lim, g, genres, lower_case):
        #Load required sample sets
        paths_d = self.locate_base(base_filepath, training_path, g)
        train_d = self.load_tt_set(paths_d['filepath'], training_path, train_pg_lim, g, lower_case)
        eval_l = self.load_tt_sets(paths_d['base'], test_path, test_pg_lim, genres, lower_case)
        #Open output file
        fobj = open( paths_d['base'] + g + "_vs_all.eval", 'w' )
        fobj.write("---- for Genre= " + g + " ----\n")
        nu_lst = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8]
        format_test_l = [2] #, 3] 
        tfreq = 1
        for trm_num in keep_term_lst:  
            self.svm_evalon_tset(fobj, train_d['tf-dict'],\
                                 train_d['tt-set'][0], train_d['tt-set'][1],\
                                 eval_l[0], eval_l[1],\
                                 g, genres,\
                                 format_test_l, nu_lst, tfreq,\
                                 trm_num )
        fobj.close()
        return ('Experiments for %s Done' % g)
    
    def exprmt_ocsvm(self, nu, featr_num, vect_format, base_filepath, training_path, train_pg_lim, test_path, test_pg_lim, g, genres, lower_case):
        #Load required sample sets
        paths_d = self.locate_base(base_filepath, training_path, g)
        train_d = self.load_tt_set(paths_d['filepath'], training_path, train_pg_lim, g, lower_case)
        eval_l = self.load_tt_sets(paths_d['base'], test_path, test_pg_lim, genres, lower_case)
        #Open output file
        fobj = open( paths_d['base'] + g + "_vs_all.eval", 'w' )
        fobj.write("---- for Genre= " + g + " ----\n")
        nu_lst = [ nu ]
        format_l = [ vect_format ] 
        tfreq = 1
        print train_d
        for trm_num in [ featr_num ]:  
            self.svm_evalon_tset(fobj, train_d['tf-dict'],\
                                 train_d['tt-set'][0], train_d['tt-set'][1],\
                                 eval_l[0], eval_l[1],\
                                 g, genres,\
                                 format_l, nu_lst, tfreq,\
                                 trm_num )
        fobj.close()
        return ('Experiments for %s Done' % g)
    
    def svm_evalon_tset(self, fobj, tf_dict, train_wpg_l, train_tf_d_l, test_wpg_l, test_tf_d_l, g, genres, format_l, nu_lst, freq_th=1, keep_terms=None):
        #If there is a limit to the terms of the dictionary that has to be kept
        if keep_terms:
            tf_dict = VHTools.keep_atleast(tf_dict, keep_terms)
            if self.keep_atleast_history == len(tf_dict):
                self.keep_atleast_history = len(tf_dict)
                return
            self.keep_atleast_history = len(tf_dict)
            fobj.write("^^^^ Terms kept= " + str(len(tf_dict)) + " ^^^^\n")
            fobj.write("TF_dictionary= " + str(tf_dict) + "\n")
        print("TF Dictionary kept length: %s" % len(tf_dict))
        #Convert TF Dictionary to Index Dictionary
        print("Convert TF Dictionary to Index Dictionary") 
        tidx_d = VHTools.tf2tidx(tf_dict)
        print('Training IdxF Dictionary size =  %s' % len(tidx_d))
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Training TF Vectors to IdxF Vectors")
        train_tf_d_l = VHTools.tf2idxf(train_tf_d_l, tidx_d)
        train_idxf_d_l = train_tf_d_l
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Test TF Vectors to IdxF Vectors")
        test_tf_d_l = VHTools.tf2idxf(test_tf_d_l, tidx_d)
        test_idxf_d_l = test_tf_d_l
        #Building a Index Frequency Dictionary 
        print("Building a Index Frequency Dictionary for this Genre: %s" % g) 
        idxf_d = VHTools.tf2idxf(tf_dict, tidx_d)
        #Start the experiments
        for i in format_l:
            if i == 1:
                fobj.write("**** Inverse Binary ****\n")
                train_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=freq_th)
                test_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(test_idxf_d_l), idxf_d, tf_threshold=freq_th)
            elif i == 2:
                fobj.write("**** Binary ****\n")
                train_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=freq_th)
                test_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(test_idxf_d_l), idxf_d, tf_threshold=freq_th)
            elif i == 3:
                fobj.write("**** Normalised by Max Term ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=True)
                test_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(test_idxf_d_l), div_by_max=True)
            elif i == 4:
                fobj.write("**** Normalised by Total Sum ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=False)             
                test_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(test_idxf_d_l), div_by_max=False)
            elif i == 5:
                fobj.write("**** Training Corpus Dictionary Frequency in TF Vectors ****\n")
                train_idxf_d_l_frmd = vformat.tf2tf_idf(copy.deepcopy(train_idxf_d_l), idxf_d)
                test_idxf_d_l_frmd = vformat.tf2tf_idf(copy.deepcopy(test_idxf_d_l), idxf_d)
            for nu in nu_lst: #[0.2, 0.3, 0.5, 0.7, 0.8]: # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                #Train SVM 
                fobj.write("++++ for nu= " + str(nu) + " ++++\n")
                print "Training"
                #for size in range(1500,9000,500):
                svm_m = SVMTE.train_oneclass_svm(fobj, train_idxf_d_l_frmd[0:], nu )
                eval_res_d = SVMTE.evaluate_oneclass_svm(svm_m, test_idxf_d_l_frmd, folds=len(genres), fld_seq_num=genres.index(g)) #len(genres))
                s = "+ %s, - %s\n" % (eval_res_d['c1'], eval_res_d['c2'])
                s += "tp=%s, tn=%s\nfp=%s, fn=%s\n" % (eval_res_d['tp'],eval_res_d['tn'],eval_res_d['fp'],eval_res_d['fn'])
                s += "Precision=%f\n" % eval_res_d['precision'] 
                s += "Recall=%f\n" % eval_res_d['recall'] 
                s += "F1=%s\n\n" % eval_res_d['f1']
                fobj.write(s)
                SVMTE.negative_indices(fobj, eval_res_d['p_labels'], len(genres))
    
    def kfold_corss_v_featr_nu(self, kfolds, keep_term_lst, base_filepath, training_path, train_pg_lim, nu_lst, g, lower_case):
        #Load required sample sets
        paths_d = self.locate_base(base_filepath, training_path, g)
        train_d = self.load_tt_set(paths_d['filepath'], training_path, train_pg_lim, g, lower_case)
        #Open output file
        fobj = open( paths_d['base'] + g + "_k-fold_cross-validation.res", 'w' )
        fobj.write("---- for Genre= " + g + " ----\n")
        fobj.write("---- folds= " + str(kfolds) + " ----\n")
        format_test_l = [2, 3] 
        tfreq = 1
        for trm_num in keep_term_lst:  
            self.svm_kfold_cross_val(kfolds,\
                                     fobj, train_d['tf-dict'],\
                                     train_d['tt-set'][0], train_d['tt-set'][1],\
                                     g,\
                                     format_test_l, nu_lst, tfreq,\
                                     trm_num )
        fobj.close()
        return ('Experiments for %s Done' % g)
    
    def svm_kfold_cross_val(self, kfolds, fobj, tf_dict, train_wpg_l, train_tf_d_l, g, format_test_l, nu_lst, freq_th=1, keep_terms=None):
        #If there is a limit to the terms of the dictionary that has to be kept
        if keep_terms:
            tf_dict = VHTools.keep_atleast(tf_dict, keep_terms)
            if self.keep_atleast_history == len(tf_dict):
                self.keep_atleast_history = len(tf_dict)
                print "REPEAT EXPERIMENT PREVENTED"
                return
            self.keep_atleast_history = len(tf_dict)
            fobj.write("^^^^ Terms kept= " + str(len(tf_dict)) + " ^^^^\n")
            fobj.write("TF_dictionary= " + str(tf_dict) + "\n")
        print("TF Dictionary kept length: %s" % len(tf_dict))
        #Convert TF Dictionary to Index Dictionary
        print("Convert TF Dictionary to Index Dictionary") 
        tidx_d = VHTools.tf2tidx(tf_dict)
        print('Training IdxF Dictionary size =  %s' % len(tidx_d))
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Training TF Vectors to IdxF Vectors")
        train_tf_d_l = VHTools.tf2idxf(train_tf_d_l, tidx_d)
        train_idxf_d_l = train_tf_d_l
        #Building a Index Frequency Dictionary 
        print("Building a Index Frequency Dictionary for this Genre: %s" % g) 
        idxf_d = VHTools.tf2idxf(tf_dict, tidx_d)
        #Start the experiments
        for i in format_test_l:
            if i == 1:
                fobj.write("**** Inverse Binary ****\n")
                train_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=freq_th)
            elif i == 2:
                fobj.write("**** Binary ****\n")
                train_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=freq_th)
            elif i == 3:
                fobj.write("**** Normalised by Max Term ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=True)
            elif i == 4:
                fobj.write("**** Normalised by Total Sum ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=False)             
            elif i == 5:
                fobj.write("**** Training Corpus Dictionary Frequency in TF Vectors ****\n")
                train_idxf_d_l_frmd = vformat.tf2tf_idf(copy.deepcopy(train_idxf_d_l), idxf_d)
            for nu in nu_lst: 
                #Train SVM 
                fobj.write("++++ for nu= " + str(nu) + " ++++\n")
                print "Performing " + str(kfolds) + "-fold cross validation for nu=" + str(nu) + " and terms=" + str(keep_terms)
                smpl_per_fld = int( math.floor( len(train_tf_d_l)/kfolds ) )
                evl_mtrx_recl = [0]*kfolds
                start = 0
                end = smpl_per_fld
                for i in range(kfolds):
                    inv_lst = self.complementof_list(train_idxf_d_l_frmd, start, end)
                    svm_m = SVMTE.train_oneclass_svm(fobj, inv_lst, nu ) #train_idxf_d_l_frmd[start:end]
                    eval_res_d = SVMTE.evaluate_oneclass_svm(svm_m, train_idxf_d_l_frmd[start:end], folds=kfolds, fld_seq_num=i) 
                    evl_mtrx_recl[ i ] = float( eval_res_d['recall'] )
                    start = end
                    end = end + smpl_per_fld    
                s = "Recall(mean)=%f\n\n" % (math.fsum(evl_mtrx_recl)/float(kfolds)) 
                fobj.write(s)
    
    def complementof_list(self, lst, excld_dwn_lim, excld_up_lim):
        if excld_dwn_lim == 0:
            return lst[excld_up_lim:]
        if excld_up_lim == len(lst):
            return lst[0:excld_dwn_lim]
        inv_lst = lst[0:excld_dwn_lim] + lst[excld_up_lim:]
        return inv_lst
    
    def svm_multiclass_featrs(self, keep_term_lst, base_filepath, training_path, train_pg_lim, test_path, test_pg_lim, c_lst, genres, lower_case):
        #Load required sample sets
        #Load Training Vectors
        train_tf_d_l = list()
        for g in genres:
            paths_d = self.locate_base(base_filepath, training_path, g)
            train_d = self.load_tt_set(paths_d['filepath'], training_path, train_pg_lim, g, lower_case)
            train_tf_d_l.extend( train_d['tt-set'][1] )
            print("Load Training Vectors for: %s" % g)
            print paths_d['filepath'] + training_path
        #Load Test Vectors
        eval_l = self.load_tt_sets(paths_d['base'], test_path, test_pg_lim, genres, lower_case)
        test_tf_d_l = eval_l[1]
        print("Load Test Vectors")
        #Generate Corpus Dictionary
        tf_dict = VHTools.gen_tfd_frmlist(train_tf_d_l)         
        #Open output file
        fobj = open( paths_d['base'] + "multiclass_svm.res", 'w' )
        format_test_l = [2, 3] 
        tfreq = 1
        for trm_num in keep_term_lst:  
            self.svm_multiclass(fobj, genres,\
                                tf_dict,\
                                train_tf_d_l, test_tf_d_l,\
                                format_test_l, tfreq,\
                                c_lst,\
                                trm_num )
        fobj.close()
        return ('Multi-class SVM Done')
    
    def svm_multiclass(self, fobj, genres, tf_dict, train_tf_d_l, test_tf_d_l, format_test_l, tfreq, c_lst, keep_terms):
        #If there is a limit to the terms of the dictionary that has to be kept
        if keep_terms:
            tf_dict = VHTools.keep_atleast(tf_dict, keep_terms)
            fobj.write("^^^^ Terms kept= " + str(len(tf_dict)) + " ^^^^\n")
            #fobj.write("TF_dictionary= " + str(tf_dict) + "\n")
        #Convert TF Dictionary to Index Dictionary
        print("Convert TF Dictionary to Term-Index Dictionary") 
        tidx_d = VHTools.tf2tidx(tf_dict)
        print('Training IdxF Dictionary size =  %s' % len(tidx_d))
        print('Test TF Vectors list size =  %s ' % len(test_tf_d_l))
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Training TF Vectors to IdxF Vectors")
        train_tf_d_l = VHTools.tf2idxf(train_tf_d_l, tidx_d)
        train_idxf_d_l = train_tf_d_l
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Test TF Vectors to IdxF Vectors")
        test_tf_d_l = VHTools.tf2idxf(test_tf_d_l, tidx_d)
        test_idxf_d_l = test_tf_d_l
        #Building a Index Frequency Dictionary 
        print("Building a Index-Frequency Dictionary for Corpus") 
        idxf_d = VHTools.tf2idxf(tf_dict, tidx_d)
        for i in format_test_l:
            if i == 1:
                fobj.write("**** Inverse Binary ****\n")
                train_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=tfreq)
                test_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(test_idxf_d_l), idxf_d, tf_threshold=tfreq)
            elif i == 2:
                fobj.write("**** Binary ****\n")
                train_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=tfreq)
                test_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(test_idxf_d_l), idxf_d, tf_threshold=tfreq)
            elif i == 3:
                fobj.write("**** Normalised by Max Term ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=True)
                test_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(test_idxf_d_l), div_by_max=True)
            elif i == 4:
                fobj.write("**** Normalised by Total Sum ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=False)             
                test_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(test_idxf_d_l), div_by_max=False)
            for c in c_lst:
                fobj.write("++++ for C= " + str(c) + " ++++\n")
                print "Training-multiclass"
                train_class_tags = list()
                for i in range( len(genres) ):
                    #print len([i+1]*(len(train_idxf_d_l_frmd)/len(genres))), i+1
                    train_class_tags.extend( [i+1]*(len(train_idxf_d_l_frmd)/len(genres)) )
                svm_m = SVMTE.train_multiclass_svm(fobj, train_idxf_d_l_frmd[0:], c, train_class_tags )
                test_class_tags = list()
                for i in range( len(genres) ):
                    #print len([i+1]*(len(test_idxf_d_l_frmd)/len(genres))), i+1
                    test_class_tags.extend( [i+1]*(len(test_idxf_d_l_frmd)/len(genres)) )
                print "Evaluating-multiclass"
                eval_res_d = SVMTE.evaluate_multiclass_svm(svm_m, test_class_tags, test_idxf_d_l_frmd, len(genres))
                s = "++++ Accuracy=%s ++++\n\n" % eval_res_d['accuracy']
                fobj.write(s)
                rl_evl_matrix = eval_res_d['eval_matrix']                        
                for i in range(len(genres)):
                    fobj.write( str( rl_evl_matrix[ i+1 ][1:] ) + "\n" )
                #fobj.write( "\n" )
                #fobj.write( str(eval_res_d['p_labels']) )
                fobj.write( "\n" )         
    
    def kfold_corss_multiclass(self, kfolds, keep_term_lst, c_lst, base_filepath, training_path, train_pg_lim, test_path, test_pg_lim, genres, lower_case):
        #Load required sample sets
        #Load Training Vectors
        tt_tf_d_ll = list()
        tt_wps_ll = list()
        class_tags_ll = list()
        tags_per_cls = list()
        tt_tf_d_l_tmp = list()
        for cls_no, g in enumerate(genres):
            paths_d = self.locate_base(base_filepath, training_path, g)
            print("Load Vectors for: %s" % g)
            print paths_d['filepath'] + training_path
            print paths_d['filepath'] + test_path
            train_d = self.load_tt_set(paths_d['filepath'], training_path, train_pg_lim, g, lower_case)
            #test_l = self.load_tt_set(paths_d['filepath'], test_path, test_pg_lim, g, lower_case) ### TEMP CHANGE
            genre_tt_tf_d_l = train_d['tt-set'][1] #+ test_l['tt-set'][1] ### TEMP CHANGE
            tt_tf_d_ll.append( genre_tt_tf_d_l  )
            tt_tf_d_l_tmp.extend( genre_tt_tf_d_l )
            tt_wps_ll.append( train_d['tt-set'][0] ) #+ test_l['tt-set'][0] ) ### TEMP CHANGE
            class_tags_ll.append( [cls_no]*len(genre_tt_tf_d_l) )
            tags_per_cls.append( len(genre_tt_tf_d_l) )
        #Generate Corpus Dictionary
        tf_dict = VHTools.gen_tfd_frmlist(tt_tf_d_l_tmp)
        del tt_tf_d_l_tmp
        #Open output file
        fobj = open( paths_d['base'] + "_k-fold_coss-val_multiclass_svm.res", 'w' )
        format_test_l = [2] #, 3] 
        tfreq = 1
        for trm_num in keep_term_lst:  
            self.multi_svm_kfold_cross_val(kfolds, fobj,\
                                           tf_dict,\
                                           tt_wps_ll, tt_tf_d_ll,\
                                           class_tags_ll, tags_per_cls,\
                                           format_test_l, c_lst, tfreq,\
                                           trm_num )
        fobj.close()
        return ('%s-fold Cross-Validation Multi-class Done' % kfolds)
    
    def multi_svm_kfold_cross_val(self, kfolds, fobj, tf_dict, tt_wpg_ll, tt_tf_d_ll, class_tags_ll, tags_per_cls, format_test_l, c_lst, freq_th=1, keep_terms=None):
        #If there is a limit to the terms of the dictionary that has to be kept
        if keep_terms:
            tf_dict = VHTools.keep_atleast(tf_dict, keep_terms)
            if self.keep_atleast_history == len(tf_dict):
                self.keep_atleast_history = len(tf_dict)
                return
            fobj.write("^^^^ Terms kept= " + str(len(tf_dict)) + " ^^^^\n")
            #fobj.write("TF_dictionary= " + str(tf_dict) + "\n")
        print("TF Dictionary kept length: %s" % len(tf_dict))
        #Convert TF Dictionary to Index Dictionary
        print("Convert TF Dictionary to Index Dictionary") 
        tidx_d = VHTools.tf2tidx(tf_dict)
        print('Training IdxF Dictionary size =  %s' % len(tidx_d))
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting TF Vectors to IdxF Vectors")
        tt_idxf_d_ll = list()
        for tt_tf_d_l in tt_tf_d_ll:
            tt_idxf_d_ll.append( VHTools.tf2idxf(tt_tf_d_l, tidx_d) )
        tt_tf_d_l = tt_idxf_d_ll
        #Building a Index Frequency Dictionary 
        print("Building a Index-Frequency Dictionary") 
        idxf_d = VHTools.tf2idxf(tf_dict, tidx_d)
        #Start the experiments
        for i in format_test_l:
            if i == 1:
                fobj.write("**** Inverse Binary ****\n")
                tt_idxf_d_ll_frmd = list()
                for tt_idxf_d_l in tt_idxf_d_ll:
                    tt_idxf_d_ll_frmd.append( vformat.inv_tf2bin(copy.deepcopy(tt_idxf_d_l), idxf_d, tf_threshold=freq_th) )
            elif i == 2:
                fobj.write("**** Binary ****\n")
                tt_idxf_d_ll_frmd = list()
                for tt_idxf_d_l in tt_idxf_d_ll:
                    tt_idxf_d_ll_frmd.append( vformat.tf2bin(copy.deepcopy(tt_idxf_d_l), idxf_d, tf_threshold=freq_th) )
            elif i == 3:
                fobj.write("**** Normalised by Max Term ****\n")
                tt_idxf_d_ll_frmd = list()
                for tt_idxf_d_l in tt_idxf_d_ll:
                    tt_idxf_d_ll_frmd.append( vformat.tf2tfnorm(copy.deepcopy(tt_idxf_d_l), div_by_max=True) )
            elif i == 4:
                fobj.write("**** Normalised by Total Sum ****\n")
                tt_idxf_d_ll_frmd = list()
                for tt_idxf_d_l in tt_idxf_d_ll:
                    tt_idxf_d_ll_frmd.append( vformat.tf2tfnorm(copy.deepcopy(tt_idxf_d_l), div_by_max=False) )             
            elif i == 5:
                fobj.write("**** Training Corpus Dictionary Frequency in TF Vectors ****\n")
                tt_idxf_d_ll_frmd = list()
                for tt_idxf_d_l in tt_idxf_d_ll:
                    tt_idxf_d_ll_frmd.append( vformat.tf2tf_idf(copy.deepcopy(tt_idxf_d_l), idxf_d) )
            ###
            kflds_sz_per_g = [ math.floor(tag_num/kfolds) for tag_num in tags_per_cls ]
            kflds_sz_residlz = [ (tag_num/float(kfolds)) - kflds_sz for tag_num, kflds_sz in zip(tags_per_cls,kflds_sz_per_g) ]
            g_lims_l = list()
            for tags_num, kflds_sz, rezs in zip(tags_per_cls, kflds_sz_per_g, kflds_sz_residlz):
                if rezs: 
                    g_lims_l.append( range( 0, tags_num, int(kflds_sz) ) )
                else:
                    g_lims_l.append( range( 0, tags_num + int(kflds_sz), int(kflds_sz) ) )
            for c in c_lst:
                fobj.write("++++ for C= " + str(c) + " ++++\n")
                print "Performing " + str(kfolds) + "-fold cross validation for C=" + str(c) + " and terms=" + str(len(tf_dict))
                evl_mtrx_recl = [0]*kfolds
                for i in range(kfolds):
                    train_inv_lst = list()
                    train_class_tags = list()
                    test_lst = list()
                    test_class_tags = list()
                    for g_lims, tt_idxf_d_l_frmd, class_tags_l in zip(g_lims_l, tt_idxf_d_ll_frmd, class_tags_ll): #tags_per_cls has so many records as the number of Genre
                        train_inv_lst.extend( self.complementof_list(tt_idxf_d_l_frmd, g_lims[i], g_lims[i+1] ) )
                        train_class_tags.extend( self.complementof_list(class_tags_l, g_lims[i], g_lims[i+1] ) )
                    svm_m = SVMTE.train_multiclass_svm(fobj, train_inv_lst, c, train_class_tags )
                    for g_lims, tt_idxf_d_l_frmd, class_tags_l in zip(g_lims_l, tt_idxf_d_ll_frmd, class_tags_ll): #tags_per_cls has so many records as the number of Genre
                        test_lst.extend( tt_idxf_d_l_frmd[ g_lims[i] : g_lims[i+1] ])
                        test_class_tags.extend( class_tags_l[ g_lims[i] : g_lims[i+1] ] )
                    eval_res_d = SVMTE.evaluate_kfold_multiclass_svm(svm_m, test_class_tags, test_lst)
                    evl_mtrx_recl[ i ] = eval_res_d
                s = "++++ Results ++++\n"
                s += "Acc.Table = " + str(evl_mtrx_recl) + "\n"    
                s += "Max.Acc. =%s\n\n"  % (math.fsum(evl_mtrx_recl)/float(kfolds))
                fobj.write(s)                   
             
"""

if __name__=='__main__':
    
    exp = SVMExperiments_with_tables()
    kfold = 10
    nu_lst = [0.1, 0.5]
    featr_size_lst = [10000]
    #fileh = tb.openFile('/home/dimitrios/Synergy-Crawler/Santini_corpus/Santini_corpus.h5', 'r+')
    fileh = tb.openFile('/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/ACC.h5', 'r+')
    #res_table = fileh.getNode('/Santini_corpus', 'Res_10FldC_3grams_Freq_OCSVM') 
    res_table =  fileh.getNode('/Automated_Crawled_Corpus', 'Res_10FldC_3grams_Freq_OCSVM')
    #res_table =  fileh.createTable('/Automated_Crawled_Corpus', 'Res_10FldC_3grams_Freq_OCSVM', ResaultsTable_desc)
    base_tbgroup = "/Automated_Crawled_Corpus/trigrams/"
    #base_tbgroup = "/Santini_corpus/trigrams/"
    genres = [ "blogs", "forum", "news", "product_pages", "wiki_pages" ] 
    #genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
    for g in genres:
        exp.OCSVM_kFold_Cross(kfold, nu_lst, featr_size_lst, fileh, base_tbgroup, res_table, g, genres)
    fileh.close()

    
    
    
    
    
    
    
    
    
    
    
    

