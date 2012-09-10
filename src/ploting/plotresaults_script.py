"""

"""
import codecs
import re
import os
import matplotlib.pyplot as plt

def load_resaults(filepath, filename):
    try:
        f = codecs.open( filepath + str(filename), "r")
    except IOError, e:
        print("FILE %s ERROR: %s" % (filename,e))
        return None
    nu_re = re.compile(r'\+\+\+\+ for nu= ([^::alpha::]+) \+\+\+\+')
    trn_vect_re = re.compile(r'#### Frequency threshold=([^::alpha::]+) ####')
    #trn_vect_re = re.compile(r'Amount of Vector for training= ([^::alpha::]+)')
    f1_re = re.compile(r'F1=([^::alpha::]+)')
    prec_re = re.compile(r'Precision=([^::alpha::]+)')
    recl_re = re.compile(r'Recall=([^::alpha::]+)')
    vformat_re =  re.compile(r'\*\*\*\*')
    titles = list() 
    vformat_d = dict()
    try:
        for lnum, fileline in enumerate(f):
            if lnum < 1:
                line = fileline.replace('\n', '')
                titles.append(line)
            if vformat_re.findall(fileline):
                tvect_d = dict(list())
                f1_d = dict(list())
                prec_d = dict(list())
                recl_d = dict(list())
                line = fileline.replace('\n', '')
                vformat_d[ line ] = (tvect_d, f1_d, prec_d, recl_d)
            next_nu = nu_re.findall(fileline)
            if next_nu:
                nu = next_nu[0]
            trn_vect = trn_vect_re.findall(fileline)
            if trn_vect:
                if nu in tvect_d: 
                    tvect_d[ nu ].append( float(trn_vect[0]) )
                else:
                    tvect_d[ nu ] = [ float(trn_vect[0]) ]
            prec = prec_re.findall( fileline )
            if prec:
                if nu in prec_d: 
                    prec_d[ nu ].append( float(prec[0]) )
                else:
                    prec_d[ nu ] = [ float(prec[0]) ]
            recl = recl_re.findall( fileline )
            if recl:
                if nu in recl_d: 
                    recl_d[ nu ].append( float(recl[0]) )
                else:
                    recl_d[ nu ] = [ float(recl[0]) ]
            f1 = f1_re.findall(fileline)
            if f1:
                if nu in f1_d: 
                    f1_d[ nu ].append( float(f1[0]) )
                else:
                    f1_d[ nu ] = [ float(f1[0]) ]        
    except Exception as e:
        print(e)
        return None
    finally:
        f.close()
    #Return the TF Vector    
    return (titles, vformat_d)

def load_resaults_2(filepath, filename):
    try:
        f = codecs.open( filepath + str(filename), "r")
    except IOError, e:
        print("FILE %s ERROR: %s" % (filename,e))
        return None
    feature_len_re = re.compile(r'\^\^\^\^ Terms kept= ([^::alpha::]+) \^\^\^\^')
    nu_re = re.compile(r'\+\+\+\+ for nu= ([^::alpha::]+) \+\+\+\+')
    #nu_re = re.compile(r'\+\+\+\+ for C= ([^::alpha::]+) \+\+\+\+')
    #f1_re = re.compile(r'F1\(mean\)=([^::alpha::]+)')
    #prec_re = re.compile(r'Precision\(mean\)=([^::alpha::]+)')
    #recl_re = re.compile(r'Recall\(mean\)=([^::alpha::]+)')
    f1_re = re.compile(r'F1=([^::alpha::]+)')
    prec_re = re.compile(r'Precision=([^::alpha::]+)')
    recl_re = re.compile(r'Recall=([^::alpha::]+)')
    accuracy_re = re.compile(r'\+\+\+\+ Accuracy=([^::alpha::]+) \+\+\+\+')
    vformat_re =  re.compile(r'\*\*\*\* ')
    titles = list() 
    vformat_d = dict()
    try:
        flagnu = False
        feature_len_val = None
        for lnum, fileline in enumerate(f):
            if lnum < 1:
                line = fileline.replace('\n', '')
                titles.append(line)
            feature_len = feature_len_re.findall(fileline)
            if feature_len:
                feature_len_val = feature_len[0]
            if vformat_re.findall(fileline):
                line = fileline.replace('\n', '')
                if line not in vformat_d:
                    orig_feature_d = dict(list())
                    orig_f1_d = dict(list())
                    orig_prec_d = dict(list())
                    orig_recl_d = dict(list())
                    orig_acc_d = dict(list())
                    vformat_d[ line ] = (orig_feature_d, orig_f1_d, orig_prec_d, orig_recl_d, orig_acc_d)
                    (feature_d, f1_d, prec_d, recl_d, acc_d) = vformat_d[ line ]
                else:
                    (feature_d, f1_d, prec_d, recl_d, acc_d) = vformat_d[ line ]
            #if feature_len:
            #    if nu in feature_d: 
            #        print nu
            #        feature_d[ nu ].append( float(feature_len[0]) )
            #    else:
            #        feature_d[ nu ] = [ float(feature_len[0]) ]
            next_nu = nu_re.findall(fileline)
            if next_nu:
                nu = next_nu[0]
                flagnu = True
            else:
                flagnu = False
            #    if feature_len_val:
            #        feature_d[ nu ] = [ float(feature_len_val) ]
            if feature_len_val and flagnu:    
                if nu in feature_d:
                    #if float(feature_len_val) not in feature_d[ nu ]:
                    feature_d[ nu ].append( float(feature_len_val) )
                else:
                    feature_d[ nu ] = [ float(feature_len_val) ]
            prec = prec_re.findall( fileline )
            if prec:
                if nu in prec_d: 
                    prec_d[ nu ].append( float(prec[0]) )
                else:
                    prec_d[ nu ] = [ float(prec[0]) ]
            recl = recl_re.findall( fileline )
            if recl:
                if nu in recl_d: 
                    recl_d[ nu ].append( float(recl[0]) )
                else:
                    recl_d[ nu ] = [ float(recl[0]) ]
            f1 = f1_re.findall(fileline)
            if f1:
                if nu in f1_d: 
                    f1_d[ nu ].append( float(f1[0]) )
                else:
                    f1_d[ nu ] = [ float(f1[0]) ]
            acc = accuracy_re.findall(fileline)
            if acc:
                if nu in acc_d: 
                    acc_d[ nu ].append( float(acc[0]) )
                else:
                    acc_d[ nu ] = [ float(acc[0]) ]        
    except Exception as e:
        print(e)
        return None
    finally:
        f.close()
    #Return the TF Vector 
    return (titles, vformat_d)

def load_resaults_3(filepath, filename):
    try:
        f = codecs.open( filepath + str(filename), "r")
    except IOError, e:
        print("FILE %s ERROR: %s" % (filename,e))
        return None
    nu_re = re.compile(r'\+\+\+\+ for nu= ([^::alpha::]+) \+\+\+\+')
    trn_vect_re = re.compile(r'#### Frequency threshold=([^::alpha::]+) ####')
    #trn_vect_re = re.compile(r'Amount of Vector for training= ([^::alpha::]+)')
    f1_re = re.compile(r'F1=([^::alpha::]+)')
    prec_re = re.compile(r'Precision=([^::alpha::]+)')
    recl_re = re.compile(r'Recall=([^::alpha::]+)')
    vformat_re =  re.compile(r'\*\*\*\*')
    titles = list() 
    vformat_d = dict()
    try:
        for lnum, fileline in enumerate(f):
            if lnum < 1:
                line = fileline.replace('\n', '')
                titles.append(line)
            if vformat_re.findall(fileline):
                line = fileline.replace('\n', '')
                if line not in vformat_d:
                    orig_tvect_d = dict(list())
                    orig_f1_d = dict(list())
                    orig_prec_d = dict(list())
                    orig_recl_d = dict(list())
                    vformat_d[ line ] = (orig_tvect_d , orig_f1_d, orig_prec_d, orig_recl_d)
                    (tvect_d , f1_d, prec_d, recl_d) = vformat_d[ line ]
                else:
                    (tvect_d , f1_d, prec_d, recl_d) = vformat_d[ line ]
                #tvect_d = dict(list())
                #f1_d = dict(list())
                #prec_d = dict(list())
                #recl_d = dict(list())
                #line = fileline.replace('\n', '')
                #vformat_d[ line ] = (tvect_d, f1_d, prec_d, recl_d)
            next_nu = nu_re.findall(fileline)
            if next_nu:
                nu = next_nu[0]
            trn_vect = trn_vect_re.findall(fileline)
            if trn_vect:
                if nu in tvect_d: 
                    tvect_d[ nu ].append( float(trn_vect[0]) )
                else:
                    tvect_d[ nu ] = [ float(trn_vect[0]) ]
            prec = prec_re.findall( fileline )
            if prec:
                if nu in prec_d: 
                    prec_d[ nu ].append( float(prec[0]) )
                else:
                    prec_d[ nu ] = [ float(prec[0]) ]
            recl = recl_re.findall( fileline )
            if recl:
                if nu in recl_d: 
                    recl_d[ nu ].append( float(recl[0]) )
                else:
                    recl_d[ nu ] = [ float(recl[0]) ]
            f1 = f1_re.findall(fileline)
            if f1:
                if nu in f1_d: 
                    f1_d[ nu ].append( float(f1[0]) )
                else:
                    f1_d[ nu ] = [ float(f1[0]) ]        
    except Exception as e:
        print(e)
        return None
    finally:
        f.close()
    #Return the TF Vector    
    return (titles, vformat_d)

def max_tuple(x_d, y_d):
    if x_d.keys() != y_d.keys():
        raise Exception('X and Y Dictionary should have the same Keys')
    max_ys = list()
    for param, y in y_d.items():
        max_ys.append( (max(y), param) )
    max_y, max_param = max(max_ys) 
    max_x = x_d[ max_param ][ y_d[max_param].index(max_y) ]
    return (max_y, max_x, max_param) 

def plot_resaults(titles, vformat_d, figure_num):
    color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ] #['b', 'g', 'm', 'k', 'r', 'g', 'b', 'c', 'y' ] #[ 1, 0.9, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8 ] #
    symbol = [ "*", "^", "x", "+", "o" , "*", "^", "x", "+" ]
    line_type = [ "-", "-", "-", "-", "-" , "--", "--", "--", "--" ]
    ylbl = ['F1', 'Prec', 'Recall', "Accuracy"]
    plt.figure( figure_num )
    plt_pos = 0 
    plt.title( titles[0] )
    for vformat in ['**** Binary ****']: #, '**** Normalised by Max Term ****']: #vformat_d.keys(): #['**** Inverse Binary ****'], '**** Binary ****', '**** Normalised by Max Term ****']: 
        tvect_d, f1_d, prec_d, recl_d, acc_d = vformat_d[ vformat ]
        for k, res_d in enumerate([f1_d, prec_d, recl_d]): #f1_d, prec_d, recl_d, acc_d 
            plt_pos += 1      
            plt.subplot(3,1, plt_pos )
            if k == 0:
                plt.title( titles[0].replace('----', '') + vformat.replace('****', '') )
            #plt.xlabel( 'Vects Number for Trainning' )
            plt.xlabel( 'Number of Features kept' )
            #plt.xlabel( 'Frequency Threshold' )
            plt.ylabel( ylbl[k] )
            for i, nu in enumerate(['0.05', '0.07', '0.1', '0.15', '0.2', '0.3', '0.5', '0.7', '0.8']): #['2', '10']): #['0.05', '0.07', '0.1', '0.15', '0.2', '0.3', '0.5', '0.7', '0.8']): #['2', '10']): # ['1', '2', '5', '10', '50']): '1', '2', '5', '10', '50']): #[0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8]): #[0.2, 0.3, 0.5, 0.7, 0.8]): 
                #print len(tvect_d[nu]), len(res_d[nu]), res_d[nu]
                plt.plot(tvect_d[ nu ], res_d[ nu ], color[i] + symbol[i] + line_type[i], label=nu) #, tvect_d[ nu ], res_d[ nu ], color[i] + line_type[i], label=nu)
                plt.legend(loc=4 )
                plt.grid(True)
            plt.Figure()
    return plt

if __name__ == '__main__':
    base_filepath = "/home/dimitrios/Synergy-Crawler/Santini_corpus/" #Experiment_th1_featrs10-100-step10_3grams_2500-500pgs_ascii_6-April-2011/z/" #cross-val_site-fold_terms/" #mutliclass_svm/" "
    flist = [files for path, dirs, files in os.walk(base_filepath)]
    flist = flist[0]
    plot_l = list()
    for z, file in enumerate(flist):
        print file
        titles, vformat_d = load_resaults_2(base_filepath, file)
        #print vformat_d['**** Binary ****']
        print "Binary", max_tuple(vformat_d['**** Binary ****'][0], vformat_d['**** Binary ****'][1])
        print "TF Normalised by Max Term", max_tuple(vformat_d['**** Normalised by Max Term ****'][0], vformat_d['**** Normalised by Max Term ****'][1])
        plt = plot_resaults(titles, vformat_d, z)
        plot_l.append( plt )
    for plt in plot_l:
        plt.show()
    
    
    
    
    
    
    
    
    
       