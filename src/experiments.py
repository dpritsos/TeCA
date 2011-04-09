"""

"""
import os
print 'os imported'
from vectorhandlingtools import VHTools
print 'VGEN IMPORTED' 
from svmutil import *
print 'SVM IMPORTED'
from trainevaloneclssvm import SVMTE
from trainevaloneclssvm import TermVectorFormat as vformat
import copy
 

class SVMExperiments(object):
    
    def __init__(self):
        self.svm = SVMTE()
    
    def tf_experiment_set1(self, base_filepath, training_path, train_pg_lim, test_path, test_pg_lim, g, genres, lower_case, fmethod="w", keep_terms=None):
        #Load TF Dictionary
        for base in base_filepath:
            filepath = base + g
            dict_filepath = base + g + training_path    
            flist = [files for path, dirs, files in os.walk(dict_filepath)]
            if flist:
                break
        print filepath
        fobj = open( base + g + "_vs_all.eval", fmethod )
        fobj.write("---- for Genre= " + g + " ----\n")
        #Load Training Vectors
        print("Load Training Vectors for: %s" % g)
        print filepath + training_path
        train_wpg_l, train_tf_d_l = VHTools.load_tfd_l_frmpaths(None, [filepath + training_path], page_lim=train_pg_lim, force_lower_case=lower_case)
        #Build Dictionary From Training Vectors
        print("Building TF Dictionary From Training Vectors")
        tf_dict = VHTools.gen_tfd_frmlist(train_tf_d_l)
        #tf_d = VHTools.load_tfd_frmpaths(None, [dict_filepath], force_lower_case=lower_case)
        #Keep atlest ### NEEDs EXPLENATION 
        if keep_terms:
            tf_dict = VHTools.keep_atleast(tf_dict, keep_terms)
            fobj.write("^^^^ Terms kept= " + str(keep_terms) + " ^^^^\n")
        print("TF Dictionary kept length: %s" % len(tf_dict))
        #Convert TF Dictionary to Index Dictionary
        print("Convert TF Dictionary to Index Dictionary") 
        tidx_d = VHTools.tf2tidx(tf_dict)
        print('Training IdxF Dictionary size =  %s' % len(tidx_d))
        #Load Test Vectors
        print("Load Test Vectors") 
        #test_wpg_l, test_tf_d_l = VHTools.load_tfd_l_frmpaths(None, [ base + g + test_path ], page_lim=test_pg_lim, force_lower_case=lower_case)
        test_wpg_l = list()
        test_tf_d_l = list()
        #test_wpg_l, test_tf_d_l =  VHTools.load_tfd_l_frmpaths(None, [ "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/blogs/test_tf_vectors/" ], page_lim=test_pg_lim, force_lower_case=lower_case) 
        #wpg, tfdl = VHTools.load_tfd_l_frmpaths(None, [ "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/news/test_tf_vectors/" ], page_lim=test_pg_lim, force_lower_case=lower_case)
        #test_wpg_l.extend( wpg ) 
        #test_tf_d_l.extend( tfdl )
        #wpg, tfdl = VHTools.load_tfd_l_frmpaths(None, [ "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/product_companies/test_tf_vectors/" ], page_lim=test_pg_lim, force_lower_case=lower_case)
        #test_wpg_l.extend( wpg ) 
        #test_tf_d_l.extend( tfdl )
        #wpg, tfdl = VHTools.load_tfd_l_frmpaths(None, [ "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/forum/test_tf_vectors/" ], page_lim=test_pg_lim, force_lower_case=lower_case)
        #test_wpg_l.extend( wpg ) 
        #test_tf_d_l.extend( tfdl )
        #wpg, tfdl = VHTools.load_tfd_l_frmpaths(None, [ "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/wiki_pages/test_tf_vectors/" ], page_lim=test_pg_lim, force_lower_case=lower_case)
        #test_wpg_l.extend( wpg ) 
        #test_tf_d_l.extend( tfdl )
        #del wpg, tfdl
        for rst_g in genres:
            wpg, tfdl = VHTools.load_tfd_l_frmpaths(None, [ base + rst_g + test_path ], page_lim=test_pg_lim, force_lower_case=lower_case) #base + rst_g + test_path
            test_wpg_l.extend( wpg ) 
            test_tf_d_l.extend( tfdl )
        del wpg, tfdl
        print('Test TF Vectors list size =  %s (%s)' % (len(test_wpg_l), len(test_tf_d_l)))
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Training TF Vectors to IdxF Vectors")
        train_tf_d_l = VHTools.tf2idxf(train_tf_d_l, tidx_d)
        train_idxf_d_l = train_tf_d_l
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Test TF Vectors to IdxF Vectors")
        test_tf_d_l = VHTools.tf2idxf(test_tf_d_l, tidx_d)
        test_idxf_d_l = test_tf_d_l
        ########################## EMPTY dict occuring when a sub-set of the Dictionary is used should be removed - Ask Professor ###########
        ###Delete IdxF Dictionary is not required any more
        ###print("Delete IdxF Dictionary")
        ###del tidx_d
        #Building a Index Frequency Dictionary 
        print("Building a Index Frequency Dictionary for this Genre: %s" % g) 
        idxf_d = VHTools.tf2idxf(tf_dict, tidx_d)
        #del tf_dict
        #del tidx_d
        #Start the experiments
        for i in [2,3]:
            ######
            TFREQ = 1
            lower_case = True
            #########Keep TF above Threshold
            #global_vect_l = tf_abv_thrld(global_vect_l, tf_threshold=TFREQ)
            #########Binary from
            if i == 1:
                fobj.write("**** Inverse Binary ****\n")
                train_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=TFREQ)
                test_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(test_idxf_d_l), idxf_d, tf_threshold=TFREQ)
            elif i == 2:
                fobj.write("**** Binary ****\n")
                train_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=TFREQ)
                test_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(test_idxf_d_l), idxf_d, tf_threshold=TFREQ)
            elif i == 3:
                fobj.write("**** Normalised by Max Term ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=True)
                test_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(test_idxf_d_l), div_by_max=True)
            elif i == 4:
                fobj.write("**** Normalised by Total Sum ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=False)             
                test_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(test_idxf_d_l), div_by_max=False)
            #########Invert TF
            #global_vect_l = inv_tf(global_vect_l) 
            #########Normalised Frequency form
            for nu in [0.2, 0.3, 0.5, 0.7, 0.8]: # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                ############################################## Train SVM ###############################################
                fobj.write("++++ for nu= " + str(nu) + " ++++\n")
                print "Training"
                #for size in range(1500,9000,500):
                svm_m = SVMTE.train_oneclass_svm(fobj, train_idxf_d_l_frmd[0:], nu )
                SVMTE.evaluate_oneclass_svm(fobj, svm_m, test_idxf_d_l_frmd, genre_no=5, g_seq_num=genres.index(g)) #len(genres))
        fobj.close()
        return ('Experiments for %s Done' % g) 
    
    def tf_experiment_set2(self, base_filepath, training_path, train_pg_lim, test_path, test_pg_lim, genres, lower_case):
        train_tf_d_l = list()
        test_tf_d_l = list()
        for g in genres:
            #Load TF Dictionary
            for base in base_filepath:
                filepath = base + g
                dict_filepath = base + g + training_path    
                flist = [files for path, dirs, files in os.walk(dict_filepath)]
                if flist:
                    break
            #Load Training Vectors
            print("Load Training Vectors for: %s" % g)
            print filepath + training_path
            train_wpg_l, train_tf_d_l_tmp = VHTools.load_tfd_l_frmpaths(None, [filepath + training_path], page_lim=train_pg_lim, force_lower_case=lower_case)
            train_tf_d_l.extend(train_tf_d_l_tmp)
            #Load Test Vectors
            print("Load Test Vectors")
            test_wpg_l, test_tf_d_l_tmp = VHTools.load_tfd_l_frmpaths(None, [ filepath + test_path ], page_lim=test_pg_lim, force_lower_case=lower_case)
            test_tf_d_l.extend(test_tf_d_l_tmp)
        #Build Dictionary From Training Vectors
        print("Building TF Dictionary From Training Vectors")
        tf_dict = VHTools.gen_tfd_frmlist(train_tf_d_l)
        #tf_d = VHTools.load_tfd_frmpaths(None, [dict_filepath], force_lower_case=lower_case)
        #Keep atlest ### NEEDs EXPLENATION 
        #tf_dict = VHTools.keep_atleast(tf_dict, 2)
        #print("TF Dictionary kept length: %s" % len(tf_dict))
        #Convert TF Dictionary to Index Dictionary
        print("Convert TF Dictionary to Term-Index Dictionary") 
        tidx_d = VHTools.tf2tidx(tf_dict)
        print('Training IdxF Dictionary size =  %s' % len(tidx_d))
        print('Test TF Vectors list size =  %s (%s)' % (len(test_wpg_l), len(test_tf_d_l)))
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Training TF Vectors to IdxF Vectors")
        train_tf_d_l = VHTools.tf2idxf(train_tf_d_l, tidx_d)
        train_idxf_d_l = train_tf_d_l
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Test TF Vectors to IdxF Vectors")
        test_tf_d_l = VHTools.tf2idxf(test_tf_d_l, tidx_d)
        test_idxf_d_l = test_tf_d_l
        ########################## EMPTY dict occuring when a sub-set of the Dictionary is used should be removed - Ask Professor ###########
        ###Delete IdxF Dictionary is not required any more
        ###print("Delete IdxF Dictionary")
        ###del tidx_d
        #Building a Index Frequency Dictionary 
        print("Building a Index Frequency Dictionary for this Genre: %s" % g) 
        idxf_d = VHTools.tf2idxf(tf_dict, tidx_d)
        #del tf_dict
        #del tidx_d
        #Start the experiments
        fobj = open( base + "multiclass_svm.eval", "w" )
        fobj.write("---- for Genre= " + g + " ----\n")
        for i in [2]:
            ######
            TFREQ = 3
            lower_case = True
            #########Keep TF above Threshold
            #global_vect_l = tf_abv_thrld(global_vect_l, tf_threshold=TFREQ)
            #########Binary from
            if i == 1:
                fobj.write("**** Inverse Binary ****\n")
                train_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=TFREQ)
                test_idxf_d_l_frmd = vformat.inv_tf2bin(copy.deepcopy(test_idxf_d_l), idxf_d, tf_threshold=TFREQ)
            elif i == 2:
                fobj.write("**** Binary ****\n")
                train_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(train_idxf_d_l), idxf_d, tf_threshold=TFREQ)
                test_idxf_d_l_frmd = vformat.tf2bin(copy.deepcopy(test_idxf_d_l), idxf_d, tf_threshold=TFREQ)
            elif i == 3:
                fobj.write("**** Normalised by Max Term ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=True)
                test_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(test_idxf_d_l), div_by_max=True)
            elif i == 4:
                fobj.write("**** Normalised by Total Sum ****\n")
                train_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(train_idxf_d_l), div_by_max=False)             
                test_idxf_d_l_frmd = vformat.tf2tfnorm(copy.deepcopy(test_idxf_d_l), div_by_max=False)
            #########Invert TF
            #global_vect_l = inv_tf(global_vect_l) 
            #########Normalised Frequency form
            for c in [2]: #, 3, 5, 7, 9]: # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                ############################################## Train SVM ###############################################
                fobj.write("++++ for C= " + str(c) + " ++++\n")
                print "Training"
                #for size in range(1500,9000,500):
                class_tags = list()
                for i in range( len(genres) ):
                    #print len([i+1]*(len(train_idxf_d_l_frmd)/len(genres))), i+1
                    class_tags.extend( [i+1]*(len(train_idxf_d_l_frmd)/len(genres)) )
                svm_m = SVMTE.train_multiclass_svm(fobj, train_idxf_d_l_frmd[0:], c, class_tags )
                test_class_tags = list()
                for i in range( len(genres) ):
                    #print len([i+1]*(len(test_idxf_d_l_frmd)/len(genres))), i+1
                    test_class_tags.extend( [i+1]*(len(test_idxf_d_l_frmd)/len(genres)) )
                SVMTE.evaluate_multiclass_svm(fobj, svm_m, test_class_tags, test_idxf_d_l_frmd, len(genres))
        fobj.close()
        return ('Experiments for %s Done' % g) 
     
    def tf_experiment_set3(self, keep_term_lims, base_filepath, training_path, train_pg_lim, test_path, test_pg_lim, g, genres, lower_case):
        start_val, upper_lim, step = keep_term_lims
        for trm_amount in range(start_val, upper_lim, step):  
            self.tf_experiment_set1(base_filepath, training_path, train_pg_lim, test_path, test_pg_lim, g, genres, lower_case, fmethod='a', keep_terms=trm_amount)
            
    def tf_experiment_set4(self, base_filepath, training_path, train_pg_lim, test_path, test_pg_lim, g, genres, freq_init, freq_lim, freq_step, lower_case, fmethod="w", keep_terms=None):
        #Load TF Dictionary
        for base in base_filepath:
            filepath = base + g
            dict_filepath = base + g + training_path    
            flist = [files for path, dirs, files in os.walk(dict_filepath)]
            if flist:
                break
        print filepath
        fobj = open( base + g + "_vs_all.eval", fmethod )
        fobj.write("---- for Genre= " + g + " ----\n")
        #Load Training Vectors
        print("Load Training Vectors for: %s" % g)
        print filepath + training_path
        train_wpg_l, train_tf_d_l = VHTools.load_tfd_l_frmpaths(None, [filepath + training_path], page_lim=train_pg_lim, force_lower_case=lower_case)
        #Build Dictionary From Training Vectors
        print("Building TF Dictionary From Training Vectors")
        tf_dict = VHTools.gen_tfd_frmlist(train_tf_d_l)
        #tf_d = VHTools.load_tfd_frmpaths(None, [dict_filepath], force_lower_case=lower_case)
        #Keep atlest ### NEEDs EXPLENATION 
        if keep_terms:
            tf_dict = VHTools.keep_atleast(tf_dict, keep_terms)
            fobj.write("^^^^ Terms kept= " + str(keep_terms) + " ^^^^\n")
        print("TF Dictionary kept length: %s" % len(tf_dict))
        #Convert TF Dictionary to Index Dictionary
        print("Convert TF Dictionary to Index Dictionary") 
        tidx_d = VHTools.tf2tidx(tf_dict)
        print('Training IdxF Dictionary size =  %s' % len(tidx_d))
        #Load Test Vectors
        print("Load Test Vectors")
        test_wpg_l, test_tf_d_l = VHTools.load_tfd_l_frmpaths(None, [ base + g + test_path ], page_lim=test_pg_lim, force_lower_case=lower_case)
        for rst_g in genres:
            if rst_g != g:
                wpg, tfdl = VHTools.load_tfd_l_frmpaths(None, [ base + rst_g + test_path ], page_lim=test_pg_lim, force_lower_case=lower_case)
                test_wpg_l.extend( wpg ) 
                test_tf_d_l.extend( tfdl )
        del wpg, tfdl
        print('Test TF Vectors list size =  %s (%s)' % (len(test_wpg_l), len(test_tf_d_l)))
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Training TF Vectors to IdxF Vectors")
        train_tf_d_l = VHTools.tf2idxf(train_tf_d_l, tidx_d)
        train_idxf_d_l = train_tf_d_l
        #Converting Training TF Vectors to IdxF Vectors
        print("Converting Test TF Vectors to IdxF Vectors")
        test_tf_d_l = VHTools.tf2idxf(test_tf_d_l, tidx_d)
        test_idxf_d_l = test_tf_d_l
        ########################## EMPTY dict occuring when a sub-set of the Dictionary is used should be removed - Ask Professor ###########
        ###Delete IdxF Dictionary is not required any more
        ###print("Delete IdxF Dictionary")
        ###del tidx_d
        #Building a Index Frequency Dictionary 
        print("Building a Index Frequency Dictionary for this Genre: %s" % g) 
        idxf_d = VHTools.tf2idxf(tf_dict, tidx_d)
        #del tf_dict
        #del tidx_d
        #Start the experiments
        lower_case = True
        for i in [2, 3]:
            for tfreq in range(freq_init, freq_lim, freq_step):
                #########Keep TF above Threshold
                #global_vect_l = tf_abv_thrld(global_vect_l, tf_threshold=TFREQ)
                #########Binary from
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
                #########Invert TF
                #global_vect_l = inv_tf(global_vect_l) 
                #########Normalised Frequency form
                for nu in [0.2, 0.5, 0.8]: #[0.2, 0.3, 0.5, 0.7, 0.8]:
                    ############################################## Train SVM ###############################################
                    fobj.write("++++ for nu= " + str(nu) + " ++++\n")
                    print "Training"
                    #for size in range(1500,9000,500):
                    fobj.write("#### Frequency threshold=" + str(tfreq) + " ####\n")
                    svm_m = SVMTE.train_oneclass_svm(fobj, train_idxf_d_l_frmd[0:], nu )
                    SVMTE.evaluate_oneclass_svm(fobj, svm_m, test_idxf_d_l_frmd, genre_no=len(genres))
        fobj.close()
        return ('Experiments for %s Done' % g)

