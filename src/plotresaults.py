"""

"""
import codecs
import re
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    f1_re = re.compile(r'F1=([^::alpha::]+)')
    prec_re = re.compile(r'Precision=([^::alpha::]+)')
    recl_re = re.compile(r'Recall=([^::alpha::]+)')
    vformat_re =  re.compile(r'\*\*\*\*')
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
                    vformat_d[ line ] = (orig_feature_d, orig_f1_d, orig_prec_d, orig_recl_d)
                    (feature_d, f1_d, prec_d, recl_d) = vformat_d[ line ]
                else:
                    (feature_d, f1_d, prec_d, recl_d) = vformat_d[ line ]
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


def plot_resaults(titles, vformat_d, figure_num):
    color = ['r', 'g', 'b', 'y', 'm']
    ylbl = ['F1', 'Prec', 'Recall']
    plt.figure( figure_num )
    plt_pos = 0 
    plt.title( titles[0] )
    for vformat in vformat_d.keys(): #['**** Inverse Binary ****', '**** Binary ****', '**** Normalised by Max Term ****']: 
        tvect_d, f1_d, prec_d, recl_d = vformat_d[ vformat ]
        for k, res_d in enumerate([f1_d, prec_d, recl_d]):
            plt_pos += 1      
            plt.subplot(3,3, plt_pos )
            plt.title( titles[0].replace('----', '') + vformat.replace('****', '') )
            #plt.xlabel( 'Vects Number for Trainning' )
            #plt.xlabel( 'Number of Features kept' )
            plt.xlabel( 'Frequency Threshold' )
            plt.ylabel( ylbl[k] )
            for i, nu in enumerate([0.2, 0.3, 0.5, 0.7, 0.8]):
                plt.plot(tvect_d[ str( nu )  ], res_d[ str( nu )  ], color[i] + 'o', tvect_d[ str( nu )  ], res_d[ str( nu )  ], color[i] + '-')
                plt.grid(True)
    return plt

if __name__ == '__main__':
    base_filepath = "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/TERMS/" #KI-04/" Santini_corpus/"
    flist = [files for path, dirs, files in os.walk(base_filepath)]
    flist = flist[0]
    plot_l = list()
    for z, file in enumerate(flist):
        print file
        titles, vformat_d = load_resaults_3(base_filepath, file)
        #print vformat_d['**** Binary ****']
        plt = plot_resaults(titles, vformat_d, z)
        plot_l.append( plt )
    for plt in plot_l:
        plt.show()
    
    
    
    
    
    
    
    
    
       