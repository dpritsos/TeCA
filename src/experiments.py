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
 

class SVMExperiments(object):
    
    def __init__(self):
        self.svm = SVMTE()
    
    def tf_experiment_set1(self, base_filepath, dict_path, vectors_path, g, genres, lower_case):
        #Load TF Dictionary
        for base in base_filepath:
            filepath = base + g
            dict_filepath = base + g + dict_path    
            flist = [files for path, dirs, files in os.walk(dict_filepath)]
            if flist:
                break
        print dict_filepath
        tf_d = VHTools.load_tfd_frmpaths(None, [dict_filepath], force_lower_case=lower_case)
        print("%s Dictionary has been loaded" % g )
        #tf_d = VHTools.keep_most_terms(tf_d, 50000)
        tf_idx_d = VHTools.tf2tidx(tf_d)                
        print('Genre Dictionary len =  %s' % len(tf_idx_d))
        #Load genre from Training and a part of it for testing
        print(filepath + vectors_path)
        wpg_l_genre, tf_d_l_genre = VHTools.load_tfd_l_frmpaths(None, [filepath + vectors_path], force_lower_case=lower_case)
        idxf_d_l_genre = VHTools.tf2idxf(tf_d_l_genre, tf_idx_d)
        train_wpg_l = wpg_l_genre[1000:] 
        train_idxf_d_l = idxf_d_l_genre[1000:]
        test_idxf_d_l = idxf_d_l_genre[0:1000]
        test_wpg_l = wpg_l_genre[0:1000] 
        #Load the rest of genres for Testing
        print("Load the rest")
        for rst_g in genres:
            if rst_g != g:
                print base + rst_g + vectors_path
                wpg, tfdl = VHTools.load_tfd_l_frmpaths(None, [ base + rst_g + vectors_path ], page_lim=1000, force_lower_case=lower_case)
                test_wpg_l.extend( wpg ) 
                test_idxf_d_l.extend( tfdl )
        del wpg, tfdl
        test_idxf_d_l = VHTools.tf2idxf(test_idxf_d_l, tf_idx_d)
        print len(test_wpg_l), len(test_idxf_d_l)
        #Start the experiments
        fobj = open( base + g + "_vs_all.eval", "w" )
        fobj.write("---- for Genre= " + g + " ----\n")
        for i in [1,2,3]:
            ######
            TFREQ = 2
            lower_case = True
            #########Keep TF above Threshold
            #global_vect_l = tf_abv_thrld(global_vect_l, tf_threshold=TFREQ)
            #########Binary from
            if i == 1:
                fobj.write("**** Inverse Binary ****\n")
                train_idxf_d_l = vformat.inv_tf2bin(train_idxf_d_l, tf_threshold=TFREQ)
                test_idxf_d_l = vformat.inv_tf2bin(test_idxf_d_l, tf_threshold=TFREQ)
            elif i == 2:
                fobj.write("**** Binary ****\n")
                train_idxf_d_l = vformat.tf2bin(train_idxf_d_l, tf_threshold=TFREQ)
                test_idxf_d_l = vformat.tf2bin(train_idxf_d_l, tf_threshold=TFREQ)
            elif i == 3:
                fobj.write("**** Normalised by Max Term ****\n")
                train_idxf_d_l = vformat.tf2tfnorm(train_idxf_d_l, div_by_max=True)
                test_idxf_d_l = vformat.tf2tfnorm(train_idxf_d_l, div_by_max=True)
            elif i == 4:
                fobj.write("**** Normalised by Total Sum ****\n")
                train_idxf_d_l = vformat.tf2tfnorm(train_idxf_d_l, div_by_max=False)             
                test_idxf_d_l = vformat.tf2tfnorm(train_idxf_d_l, div_by_max=False)
            #########Invert TF
            #global_vect_l = inv_tf(global_vect_l) 
            #########Normalised Frequency form
            for nu in [0.2, 0.3, 0.5, 0.7, 0.8]: # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                ############################################## Train SVM ###############################################
                fobj.write("++++ for nu= " + str(nu) + " ++++\n")
                print "Training"
                #for size in range(1500,9000,500):
                svm_m = SVMTE.train_svm(fobj, train_idxf_d_l[0:], nu )
                SVMTE.evaluate_svm(fobj, svm_m, test_idxf_d_l)
        fobj.close()
        return ('Experiments for %s Done' % g) 
    
    
    
    def set1(self, base_filepath, corpus_d, vectors_d, g, genres, lower_case):
        
        for base in base_filepath:
            filepath = base + g + corpus_d    
            cdicts_flist = [files for path, dirs, files in os.walk(filepath)]
            if cdicts_flist:
                cdicts_flist = cdicts_flist[0]
                corpus_dict = merge_to_global_dict(cdicts_flist, filepath, force_lower_case=lower_case)
                print("%s Dictionary has been loaded" % g )
                corpus_dict = keep_most_terms(corpus_dict, 5000)
                gterm_index = merge_global_dicts( corpus_dict )
                base_filepath = base
                break
        print 'gterms len = ', len(gterm_index)
    
        wpsl_genre = dict() 
        vectl_genre = dict()
        self.vform.glob_vect_l(base_filepath, vectors_d, [ g ] , gterm_index, None, wpsl_genre, vectl_genre, True)
        rest_genres = list()
        for rst_g in genres:
            if rst_g != g:
                rest_genres.append(rst_g)
        self.vform.glob_vect_l(base_filepath, vectors_d, rest_genres, gterm_index, 1000, wpsl_genre, vectl_genre, True)
        print len(wpsl_genre), len(vectl_genre)
        
        fobj = open( base_filepath + g + "_vs_all.eval", "w" )
        fobj.write("---- for Genre= " + g + " ----\n")
        for i in [1,2,3]:
            ######
            TFREQ = 2
            lower_case = True
            #########Keep TF above Threshold
            #global_vect_l = tf_abv_thrld(global_vect_l, tf_threshold=TFREQ)
            #########Binary from
            if i == 1:
                fobj.write("**** Inverse Binary ****\n")
                for grn in genres:
                    vectl_genre[ grn ] = self.vform.inv_tf2bin(vectl_genre[ grn ], tf_threshold=TFREQ)
            elif i == 2:
                fobj.write("**** Binary ****\n")
                for grn in genres:
                    vectl_genre[ grn ] = self.vform.tf2bin(vectl_genre[ grn ], tf_threshold=TFREQ)
            elif i == 3:
                fobj.write("**** Normilised by Max Term ****\n")
                for grn in genres:
                    vectl_genre[ grn ] = self.vform.tf2tfnorm(vectl_genre[ grn ], div_by_max=True)
            elif i == 4:
                fobj.write("**** Normalised by Total Sum ****\n")
                for grn in genres:
                    vectl_genre[ grn ] = self.vform.tf2tfnorm(vectl_genre[ grn ], div_by_max=False)             
            #########Invert TF
            #global_vect_l = inv_tf(global_vect_l) 
            #########Normalised Frequency form
            for nu in [0.2, 0.3, 0.5, 0.7, 0.8]: # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                ############################################## Train SVM ###############################################
                fobj.write("++++ for nu= " + str(nu) + " ++++\n")
                print "Training"
                for size in range(1500,9000,500):
                    class_tags, svm_m = self.svm.train_svm(fobj, vectl_genre[ g ][1000:size], nu )
                    #print("Labels %s" % svm_m.get_labels()) Not working for one-class SVM
                    ############################################### Evaluate SVM ##############################################
                    #fobj.write("---- for Genre= " + g + " ----\n")
                    self.svm.evaluate_svm(fobj, svm_m, vectl_genre, g, genres)
        fobj.close()
        return ('Experiments for %s Done' % g) 



