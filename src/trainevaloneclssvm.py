"""

"""
import os
from svmutil import *
import decimal
from vectorhandlingtools import *


class SVMTE(object): 
    
    @staticmethod
    def train_multiclass_svm(fobj, training_vectors, c, class_tags=None):
        if class_tags == None:
            class_tags = [0]*len(training_vectors)
        print(len(class_tags))
        fobj.write( "Amount of Vector for training= " + str(len(class_tags)) + "\n" ) 
        prob = svm_problem(class_tags, training_vectors)
        params_s = '-s 0 -t 0 -c ' + str(c)
        print params_s
        model_params = svm_parameter( params_s )
        svm_m = svm_train(prob, model_params)
        print("Done!")
        return svm_m
    
    @staticmethod
    def train_oneclass_svm(fobj, training_vectors, nu, class_tags=None):
        if class_tags == None:
            class_tags = [0]*len(training_vectors)
        print(len(class_tags))
        fobj.write( "Amount of Vector for training= " + str(len(class_tags)) + "\n" ) 
        prob = svm_problem(class_tags, training_vectors)
        params_s = '-h 0 -s 2 -t 0 -n ' + str(nu)
        #params_s = '-s 2 -t 0 -n ' + str(nu)
        print params_s
        model_params = svm_parameter( params_s )
        svm_m = svm_train(prob, model_params)
        print("Done!")
        return svm_m
    
    @staticmethod
    def evaluate_oneclass_svm(fobj, svm_m, test_idxf_d_l, genre_no):
        c1 = float(0)
        c2 = float(0)
        tp = float(0) 
        tn = float(0)
        fp = float(0)
        fn = float(0)
        #Prediction phase
        print len(test_idxf_d_l)
        print "Predicting"  
        p_labels, acc, val = svm_predict([0]*len(test_idxf_d_l), test_idxf_d_l, svm_m, '-b 0' )
        #print p_labels
        tg_lim = len(p_labels)/genre_no
        print "Evaluating"
        for i, d in enumerate(p_labels):
            if d > 0:
                if i > tg_lim:
                    fp += 1
                else:
                    tp += 1   
                c1 += 1
            else:
                if i > tg_lim:
                    tn += 1
                else:
                    fn += 1
                c2 += 1
        ##
        s = "+ %s, - %s\n" % (c1, c2)
        fobj.write(s)
        s = "tp=%s, tn=%s\nfp=%s, fn=%s\n" % (tp,tn,fp,fn)
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
            s = "Precision=0.0\nRecall=0.0\nF1=0.0\n\n" 
            fobj.write(s)
        #OPTIONAL FUNCTION CALL
        SVMTE.evaluate_oneclass_svm_2(fobj, p_labels, genre_no)
    
    @staticmethod
    def evaluate_oneclass_svm_2(fobj, p_labels, genre_no):
        pg_per_g = len(p_labels)/genre_no
        evl_matrix = list(range(genre_no)) 
        start = 0
        end = pg_per_g
        for i in range(genre_no):
            for p_lbl in p_labels[start:end]:
                if p_lbl == 1:
                    evl_matrix[ i ] += 1
            start = end
            end = end + pg_per_g
        for i in range(genre_no):
            fobj.write( str( evl_matrix[ i ] ) + "\n" )
        #fobj.write( str(p_labels) )
    
    @staticmethod
    def evaluate_multiclass_svm(fobj, svm_m, class_tags, test_idxf_d_l, genre_no):
        print len(test_idxf_d_l)
        print "Predicting"  
        p_labels, acc, val = svm_predict(class_tags, test_idxf_d_l, svm_m, '-b 0' )
        print "Evaluating"
        print acc[0]
        try:
            s = "Accuracy=%s\n" % acc[0]
            fobj.write(s)
        except Exception as e:
            s = "ERROR " + e
            fobj.write(s)
        pg_per_g = len(p_labels)/genre_no
        rl_evl_matrix = list()
        for i in range( genre_no + 1 ):
            rl_evl_matrix.append( [0*i for i in range( genre_no + 1 )] ) 
        start = 0
        end = pg_per_g
        for i in range(genre_no):
            for p_lbl in p_labels[start:end]:
                rl_evl_matrix[ i + 1 ][  int(p_lbl) ] += 1
            start = end
            end = end + pg_per_g
        #print rl_evl_matrix
        #print "\n\n"
        #print p_labels   
        for i in range(genre_no):
            fobj.write( str( rl_evl_matrix[ i+1 ][1:] ) + "\n" )
        fobj.write( str(p_labels) )
            
            
    @staticmethod
    def evaluate_multiclass_svm_2(fobj, svm_m, class_tags, test_idxf_d_l, genre_no):
        c1 = decimal.Decimal('0')
        c2 = decimal.Decimal('0')
        tp = decimal.Decimal('0') 
        tn = decimal.Decimal('0')
        fp = decimal.Decimal('0')
        fn = decimal.Decimal('0')
        #Prediction phase
        #print len(vectl_genre['news'][3000:4000]), len(vectl_genre['product_companies'][0:3000]), len(total_global_vect_l)
        print len(test_idxf_d_l)
        print "Predicting"  
        p_labels, acc, val = svm_predict(class_tags, test_idxf_d_l, svm_m, '-b 0' )
        print "Evaluating"
        print acc[0]
        try:
            s = "Accuracy=%s\n" % acc[0]
            fobj.write(s)
        except Exception as e:
            s = "ERROR " + e
            fobj.write(s)
        pg_per_g = len(p_labels)/genre_no
        for i in range(genre_no):
            c1 = float(0)
            c2 = float(0)
            tp = float(0) 
            tn = float(0)
            fp = float(0)
            fn = float(0)
            for n, l in enumerate(p_labels):
                if l == (i+1):
                    if n >= pg_per_g*i and n < pg_per_g*(i+1):
                        tp += 1
                    else:
                        fp += 1   
                    c1 += 1
                else:
                    if n >= pg_per_g*i and n < pg_per_g*(i+1):
                        fn += 1
                    else:
                        tn += 1
                    c2 += 1
            ##
            s = "+ %s, - %s\n" % (c1, c2)
            fobj.write(s)
            s = "tp=%s, tn=%s\nfp=%s, fn=%s\n" % (tp,tn,fp,fn)
            fobj.write(s)
    


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
                max = float( 0 )
                for key in line.keys():
                    #print "MAKZ", max
                    if line[key] > max:
                        max = line[key]
                if max > 0:
                    for key in line.keys():
                        line[key] = (line[key]/max)
                else:
                    pass
                    #print("tf2tnorm MAX<0 on list line => len %s :: line %s" % (len(line),line))
        else:
            for line in global_vect_l:
                sum = float( 0 )
                for key in line.keys():
                    sum += line[key]
                for key in line.keys():
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
    def tf2bin(tf_vect_l, tf_d, tf_threshold=0):
        for tf_vect in tf_vect_l:
            for term in tf_vect.keys():
                if tf_d[term] > tf_threshold:
                    tf_vect[term] = 1
                else:
                    tf_vect[term] = 0
        return tf_vect_l
    
    @staticmethod
    def inv_tf2bin(tf_vect_l, tf_d, tf_threshold=0):
        for tf_vect in tf_vect_l:
            for term in tf_vect.keys():
                if tf_d[term] < tf_threshold:
                    tf_vect[term] = 1
                else:
                    tf_vect[term] = 0
        return tf_vect_l

    