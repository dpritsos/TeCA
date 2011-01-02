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
    trn_vect_re = re.compile(r'Amount of Vector for training= ([^::alpha::]+)')
    f1_re = re.compile(r'F1=([^::alpha::]+)')
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
                line = fileline.replace('\n', '')
                vformat_d[ line ] = (tvect_d, f1_d)
            next_nu = nu_re.findall(fileline)
            if next_nu:
                nu = next_nu[0]
            trn_vect = trn_vect_re.findall(fileline)
            if trn_vect:
                if nu in tvect_d: 
                    tvect_d[ nu ].append( float(trn_vect[0]) )
                else:
                    tvect_d[ nu ] = [ float(trn_vect[0]) ]
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


if __name__ == '__main__':
    base_filepath = "/home/dimitrios/Documents/Synergy-Crawler/experiment_res/"
    flist = [files for path, dirs, files in os.walk(base_filepath)]
    flist = flist[0]
    titles, vformat_d = load_resaults(base_filepath, flist[0])
    print titles, vformat_d    
    plt.title( titles[0] )
    plt.xlabel( 'Vects Number for Trainning' )
    plt.ylabel('F1')
    color = ['r', 'g', 'b', 'y', 'm']
    tvect_d, f1_d = vformat_d['**** Inverse Binary ****']
    for i, nu in enumerate([0.2, 0.3, 0.5, 0.7, 0.8]):
        plt.plot(tvect_d[ str( nu )  ], f1_d[ str( nu )  ], color[i] + 'o', tvect_d[ str( nu )  ], f1_d[ str( nu )  ], color[i] + '-')
    plt.grid(True)
    plt.show()

    
    
    
    
    
    
    
    
    
    
       