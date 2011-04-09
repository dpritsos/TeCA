"""

"""

import codecs
import re
import os

def load_sets_intersected(filepath, filename):
    try:
        f = codecs.open( filepath + str(filename), "r")
    except IOError, e:
        print("FILE %s ERROR: %s" % (filename,e))
        return None
    set_re = re.compile(r'negative_indices->\[([^::alpha::]+)\]\n')
    num_re = re.compile(r'(\d+),')
    set_l = list()
    try: 
        for lnum, fileline in enumerate(f):
            set_line = set_re.findall(fileline)
            if set_line:
                num_list = num_re.findall(set_line[0])
                set_l.append( set(num_list) )
        intersected_s = set_l[0]
        for s in set_l[1:0]:
            intersected_s.intersection_update(s)    
    except Exception as e:
        print(e)
        return None
    finally:
        f.close()
    #Return the TF Vector    
    return intersected_s


if __name__ == '__main__':
    base_filepath = "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/TEST/" #Santini_corpus/"
    flist = [files for path, dirs, files in os.walk(base_filepath)]
    flist = flist[0]
    plot_l = list()
    intrscted_s_l = list()
    for z, file in enumerate(flist):
        print file
        intersected_s = load_sets_intersected(base_filepath, file)
        print file," Intersection: ",intersected_s
        print "Intersected indices amount:",len(intersected_s)
        intrscted_s_l.append(intersected_s)
    total_inter_s = intrscted_s_l[0]
    for s in intrscted_s_l[1:]:
        total_inter_s.intersection_update(s)
    print "Total intersection: ",total_inter_s 
    print "Intersected indices amount: ",len(total_inter_s)
