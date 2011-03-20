

import codecs
import re
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from vetorhandlingtools import VHTools

class Histograms(object):
    
    def tf_d_hist(self, base_filepath, training_path, test_path, g, genres, lower_case):
        #Load TF Dictionary
        for base in base_filepath:
            filepath = base + g
            dict_filepath = base + g + training_path    
            flist = [files for path, dirs, files in os.walk(dict_filepath)]
            if flist:
                break
        print filepath
        #Load Training Vectors
        print("Load Training Vectors for: %s" % g)
        print filepath + training_path
        train_wpg_l, train_tf_d_l = VHTools.load_tfd_l_frmpaths(None, [filepath + training_path], force_lower_case=lower_case)
        #Build Dictionary From Training Vectors
        print("Building TF Dictionary From Training Vectors")
        tf_dict = VHTools.gen_tfd_frmlist(train_tf_d_l)
        #tf_d = VHTools.load_tfd_frmpaths(None, [dict_filepath], force_lower_case=lower_case)
        #Keep atlest ### NEEDs EXPLENATION 
        #tf_dict = VHTools.keep_atleast(tf_dict, 99)
        #print("TF Dictionary kept length: %s" % len(tf_dict))
        #Convert TF Dictionary to Index Dictionary
        print("Convert TF Dictionary to Index Dictionary") 
        tidx_d = VHTools.tf2tidx(tf_dict)
        #Building a Index Frequency Dictionary
        print("Building a Index Frequency Dictionary for this Genre: %s" % g) 
        idxf_d = VHTools.tf2idxf(tf_dict, tidx_d)
        
        
        