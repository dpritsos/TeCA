"""

"""
import os
from svmutil import *
import decimal
from vectorhandlingtools import *


class SVMTE(object): 
    
    @staticmethod
    def train_svm(fobj, training_vectors, nu, class_tags=None):
        if class_tags == None:
            class_tags = [0]*len(training_vectors)
        print(len(class_tags))
        fobj.write( "Amount of Vector for training= " + str(len(class_tags)) + "\n" ) 
        prob = svm_problem(class_tags, training_vectors)
        params_s = '-h 0 -s 2 -t 0 -n ' + str(nu)
        print params_s
        model_params = svm_parameter( params_s )
        svm_m = svm_train(prob, model_params)
        print("Done!")
        return svm_m
    
    @staticmethod
    def evaluate_svm(fobj, svm_m, vectl_genre, trained_genre, genres):
        c1 = decimal.Decimal('0')
        c2 = decimal.Decimal('0')
        tp = decimal.Decimal('0') 
        tn = decimal.Decimal('0')
        fp = decimal.Decimal('0')
        fn = decimal.Decimal('0')
        total_global_vect_l = vectl_genre[trained_genre][0:1000]
        for g in genres:
            if g != trained_genre:
                total_global_vect_l.extend(vectl_genre[ g ][0:1000]) 
        #Prediction phase
        #print len(vectl_genre['news'][3000:4000]), len(vectl_genre['product_companies'][0:3000]), len(total_global_vect_l)
        print len(total_global_vect_l)
        print "Predicting"  
        p_labels, acc, val = svm_predict([0]*len(total_global_vect_l), total_global_vect_l, svm_m, '-b 0' )
        #print p_labels
        print "Evaluating"
        for i, d in enumerate(p_labels):
            if d > 0:
                if i > 999:
                    fp += 1
                else:
                    tp += 1   
                c1 += 1
            else:
                if i > 999:
                    tn += 1
                else:
                    fn += 1
                c2 += 1
        ##
        s = "+ %s, - %s\n" % (c1, c2)
        fobj.write(s)
        s = "tp=%s, fp=%s, tn=%s, fn=%s\n" % (tp,fp,tn,fn)
        fobj.write(s)
        try:
            ##
            precision = tp/(tp+fp)
            s = "Precision=%f\n" % precision 
            fobj.write(s)
            ##
            recall = tp/(tp+fn)
            s = "Recall=%f\n" % recall 
            fobj.write(s)
            ##
            f1 = (2*precision*recall)/(precision+recall)
            s = "F1=%s\n\n" % f1
            fobj.write(s)
        except Exception as e:
            fobj.write( str(e)+"\n" )



class TermVectorFormat(object):
    
    @staticmethod
    def tf_abv_thrld(global_vect_l, tf_threshold=0):
        for line in global_vect_l:
            dkeys = line.keys()
            for key in dkeys:
                if line[key] < tf_threshold:
                    line[key] = float( 0 )
        return global_vect_l

    @staticmethod
    def tf2tfnorm(global_vect_l, div_by_max=False):
        if div_by_max:
            for line in global_vect_l:
                dkeys = line.keys()
                max = float( 0 )
                for key in dkeys:
                    if line[key] > max:
                        max = line[key]
                if max > 0:
                    for key in dkeys:
                        line[key] = (line[key]/max)
                else:
                    pass
                    #print("tf2tnorm MAX<0 on list line => len %s :: line %s" % (len(line),line))
        else:
            for line in global_vect_l:
                dkeys = line.keys()
                sum = float( 0 )
                for key in dkeys:
                    sum += line[key]
                for key in dkeys:
                    line[key] = line[key]/sum
        return global_vect_l
    
    @staticmethod
    def inv_tf(global_vect_l):
        for line in global_vect_l:
            dkeys = line.keys()
            for key in dkeys:
                line[key] = (1/line[key])
        return global_vect_l

    @staticmethod
    def tf2bin(global_vect_l, tf_threshold=0):
        for line in global_vect_l:
            dkeys = line.keys()
            for key in dkeys:
                if line[key] > tf_threshold:
                    line[key] = 1
                else:
                    line[key] = 0
            return global_vect_l

    @staticmethod
    def inv_tf2bin(global_vect_l, tf_threshold=0):
        for line in global_vect_l:
            dkeys = line.keys()
            for key in dkeys:
                if line[key] < tf_threshold:
                    line[key] = 1
                else:
                    line[key] = 0
        return global_vect_l

    