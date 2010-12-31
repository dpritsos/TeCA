"""

"""
import codecs

def load_resaults(filepath, filename):
    try:
        f = codecs.open( filepath + str(filename), "r")
    except IOError, e:
        print("FILE %s ERROR: %s" % (filename,e))
        return None
    #The following for loop is an alternative approach to reading lines instead of using f.readline() or f.readlines()
    nu = dict()
    
    try:
        for fileline in f:
            line = fileline.replace('\n', '')
            line = line.split(" => ") #BE CAREFULL with SPACES
            
    except:
        f.close()
        return None
    f.close()
    #Return the TF Vector    
    return vect_dict 