"""

"""
import os
from vectorhandlingtools import *
from termvectorgenerator import VectGen
from svmutil import *
from trainevaloneclssvm import *
 

class SVMExperiments(object):
    
    def __init__(self):
        self.svm = SVMTE()
        self.vform = TermVectorFormat()
    
    def set1(self,base_filepath, g, vectl_genre, genres):
        fobj = open( base_filepath + g + "_vs_all.eval", "w" )
        fobj.write("---- for Genre= " + g + " ----\n")
        for i in [1,2,3]:
            ######
            TFREQ = 5
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
                fobj.write("**** Normilised by Total Sum ****\n")
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


print "Thank you and Goodbye!"

