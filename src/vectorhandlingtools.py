"""

"""

import eventlet
import codecs
import os
#from scgenrelerner_svmbased import *

class VHTools(object):

    @staticmethod
    def keep_most_terms(terms_d, terms_amout):
        """ keep_most_terms(): is getting a dictionary of Terms-Frequencies and 
            the amount of Terms to return as arguments. It is returning the number
            of Terms equal to the argument 'terms_amount' with the Highest Frequency. """
        terms_l = [(v, k) for (k, v) in terms_d.iteritems()]
        terms_l.sort()
        terms_l.reverse()
        most_terms_l = terms_l[0: terms_amout]
        terms_d = dict( [ (k, v) for (v, k) in most_terms_l ] )
        return terms_d    
        
    @staticmethod
    def merge_tf_dicts(*terms_d):
        """ mege_dicts(): is getting a set of term-frequency dictionaries as list of
            arguments and return a dictionary of common terms with their sum of frequencies
            of occurred in all dictionaries containing these terms. """
        tf_d = dict()
        tf_l = list()
        for tr_dict in terms_d:
            tf_l.extend( tr_dict.items() )    
        for i in range(len(tf_l)):
            if tf_l[i][0] in tf_d: 
                tf_d[ tf_l[i][0] ] += tf_l[i][1]
            else:
                tf_d[ tf_l[i][0] ] = tf_l[i][1]
        return tf_d
    
    @staticmethod
    def tf_dict_idxing(term_d):
        """ tf_dict_idxing(): is getting a Term-Frequency dictionary and returns one
            with terms-index number. The index number is just their position in the
            descending order sorted list of dictionary keys. """
        term_l = term_d.keys()
        term_l.sort()
        idx = range( len(term_l) + 1 )
        term_idx_d = dict( zip( term_l , idx[1:] ) )
        return term_idx_d
    
    @staticmethod
    def gener_tf_d(webpg_vect_l):
        """ gener_tf_d(): is getting a list of Term-Frequency Dictionaries and creates a 
            TF Dictionary of all terms occurred in the list. """
        return VHTools.merge_tf_dicts( *webpg_vect_l )
    
    @staticmethod
    def load_dict(filepath, filename, force_lower_case=False):
        try:
            f = codecs.open( filepath + str(filename), 'r', 'utf-8', 'strict')
        except IOError as e:
            print("FILE %s ERROR: %s" % (filename,e))
            return None
        #The following for loop is an alternative approach to reading lines instead of using f.readline() or f.readlines()
        vect_dict = dict()
        try:
            for fileline in f:
                line = fileline.replace('\n', '')
                line = line.split(" => ") #BE CAREFULL with SPACES
                if force_lower_case: 
                    vect_dict[ line[0].lower() ] = float( line[1] )
                else:
                    vect_dict[ line[0] ] = float( line[1] )
        except:
            f.close()
            return None
        f.close()
        #Return the TF Vector    
        return vect_dict  

    @staticmethod
    def merge_to_global_dict(filelist, filepath=None, force_lower_case=False):
        if not isinstance(filelist, (list, tuple)) :
            return False
        gpool = eventlet.GreenPool(10)
        filepaths= map( lambda x: filepath, range(len(filelist)) )
        force_lower= map( lambda x: force_lower_case, range(len(filelist)) )
        #Start Merging the Dictionaries - or Vector of Term Frequencies
        global_vect = VHTools.load_dict(filepath, filelist[0], force_lower_case)
        for vect_d in gpool.imap(VHTools.load_dict, filepaths, filelist[1:], force_lower):
            for d_trm in vect_d:
                if d_trm in global_vect: 
                    global_vect[d_trm] += vect_d[d_trm] 
                else:
                    global_vect[d_trm] = vect_d[d_trm]
        return global_vect
    
    @staticmethod
    def load_dict_l_Depricated(filepath, filename, g_terms_d=None, force_lower_case=False, page_num=0):
        try:
            f = codecs.open( filepath + str(filename), "r", "utf-8")
        except IOError, e:
            print("FILE %s ERROR: %s" % (filename,e))
            return None
        #The following for loop is an alternative approach to reading lines instead of using f.readline() or f.readlines()
        wps_l = list()
        vect_l = list()
        try:
            for fileline in f:
                line = fileline.split(" => ") #BE CAREFULL with SPACES
                wps_l.append( line[0] )
                composed_terms = line[1].split('\t')
                vect_dict = dict()  
                for comp_term in composed_terms:
                    if comp_term == '\n': # or comp_term == ' ':
                        continue
                    decomp_term = comp_term.split(' : ') 
                    if len(decomp_term) == 2:
                        if g_terms_d == None:
                            if force_lower_case:
                                vect_dict[ decomp_term[0].lower() ] = float( decomp_term[1] )
                            else:    
                                vect_dict[ decomp_term[0] ] = float( decomp_term[1] )
                        elif isinstance(g_terms_d, dict):
                            #if Globals Term list has been given then find the proper index value for creating the numerically tagged dictionary
                            try:
                                if force_lower_case:
                                    vect_dict[ g_terms_d[ decomp_term[0].lower() ] ] = float( decomp_term[1] )
                                else:
                                    vect_dict[ g_terms_d[ decomp_term[0] ] ] = float( decomp_term[1] )
                            except:
                                #if you cannot find the term in the global dictionary just drop the term
                                #print("Term \" %s \"not found in the Global Dictionary/Index - Dropped!" % decomp_term[0])
                                pass
                vect_l.append( vect_dict )
                #If a limited number of HTML page vector is needed then stop loading when this number is reached 
                if len(vect_l) == page_num:
                    break
        except Exception as e:
            f.close()
            print "vectorhandlingtools.load_dict_l() while reading: ", e
            return None
        finally:
            f.close()
        #Return tuple of WebPages Vectors and     
        return (wps_l, vect_l)

    @staticmethod
    def load_dict_l(filepath, filename, force_lower_case=False): #, g_terms_d=None, force_lower_case=False, page_num=0):
        try:
            f = codecs.open( filepath + str(filename), "r", "utf-8")
        except IOError, e:
            print("VHTools.load_dict_l() FILE %s ERROR: %s" % (filename,e))
            return None
        #The following for loop is an alternative approach to reading lines instead of using f.readline() or f.readlines()
        wps_l = list()
        vect_l = list()
        try:
            for fileline in f:
                wp_name, wp_tf_d = tuple( fileline.split(" => ") ) #BE CAREFULL with SPACES
                wps_l.append( wp_name )
                composed_terms = wp_tf_d.split('\t\t')
                composed_terms.replace('\n', '')
                vect_dict = dict()  
                for comp_term in composed_terms:
                    Term, Freq = tuple( comp_term.split(' : ') ) 
                    if force_lower_case:
                        vect_dict[ Term.lower() ] = float( Freq )
                    else:    
                        vect_dict[ Term ] = float( Freq )
                vect_l.append( vect_dict )
        except Exception as e:
            f.close()
            print("VHTools.load_dict_l() Error: %s" % e)
            return None
        finally:
            f.close()
        #Return tuple of WebPages Vectors and     
        return (wps_l, vect_l)

    @staticmethod
    def __tf2idxf(webpg_vect, tf_idx_d):
        """ __tf2idxf(): Don't use it directly, use tf2idxf instead.
            This function is getting a TF dictionary representing the TF Vector,
            and a TF-Index as defined in VHTools.tf_dict_idxing(). It returns
            a Index-Frequency dictionary where each term of the TF dictionary has been 
            replaced with the Index number of the TF-Index. """
        idxed_webpg_vect = dict() 
        for term, freq in webpg_vect.items():
            idxed_webpg_vect[ tf_idx_d[term] ] = freq
        return idxed_webpg_vect
    
    @staticmethod
    def tf2idxf(webpg_vect_l, tf_idx_d):
        """ tf2idxf(): is getting a TF-Dictionary or a list of TF-Dictionaries and TF-Index. It applys
            the VHTools.__tf2idxf() function to the dictionaries and returns a list or single TF-Dictioary
            depending on the input. """
        if isinstance(webpg_vect_l, list):
            idxed_webpg_vect = list()
            for webpage in webpg_vect_l:
                idxed_webpg_vect.append( VHTools.__tf2idxf(webpage, tf_idx_d) )
            return idxed_webpg_vect
        elif isinstance(webpg_vect_l, dict):
            return VHTools.__tf2idxf(webpg_vect_l, tf_idx_d)

    @staticmethod
    def save_dct(filename, records, filepath=None):
        """save_dct():"""
        try:
            #Codecs needed for saving string that are encoded in UTF8, but I do not need it because strings are already the Proper Encoding form
            f = codecs.open( filepath + filename, "w", "utf-8") #change "utf-8" to xcharset
        except IOError:
            return None 
        try: 
            for rec in records:
                f.write(rec + " => "  + str(records[rec]) + "\n") # Write a string to a file 
        except:
            print("ERROR WRITTING FILE: %s" % filename)
        finally:
            f.close()
        return True           

    @staticmethod
    def save_dct_lst(filename, records, index, filepath=None):
        try:
            #Codecs needed for saving string that are encoded in UTF8, but I do not need it because strings are already the Proper Encoding form
            print "SAVING, ", filepath + filename
            f = codecs.open( filepath + filename, "w", "utf-8") #change "utf-8" to xcharset
        except IOError as e:
            print "save_dct_lst Error: ", e
            return None 
        try: 
            for i in range(len(index)):
                f.write(index[i] + " => ")
                for rec in records[i]:
                    f.write( str(rec) + " : "  + str(records[i][rec]) + "\t") 
                f.write("\n") 
        except Exception as e:
            print("ERROR WRITTING FILE: %s -- %s" % (filename, e))
        finally:
            f.close()
        return True       

    @staticmethod
    def glob_vect_l(base_filepath, vectors_d, genres, gterm_index, pg_num=None, wpsl_genre={}, vectl_genre={}, lower_case=False):
        if pg_num == None:
            pg_num = 0
        for g in genres:
            filepath = base_filepath + g + vectors_d
            vect_flist = [files for path, dirs, files in os.walk(filepath)]
            vect_flist = vect_flist[0] 
            global_wps_l = list()
            global_vect_l = list()
            for filename in vect_flist:
                resault = VHTools.load_dict_l(filepath, filename, gterm_index, force_lower_case=lower_case, page_num=pg_num)
                if resault:
                    wps_l, vect_l = resault
                    global_wps_l.extend( wps_l )
                    global_vect_l.extend( vect_l )
                    print("%s global_vect_l len: %s" % (g, len(global_vect_l)))
            #CLEAN VECTOR LIST
            for i, line in enumerate(global_vect_l):
                if 500 and line > 1000:  ###############
                    del global_vect_l[i]
                    del global_wps_l[i]
            wpsl_genre[ g ] = global_wps_l
            vectl_genre[ g ] = global_vect_l
            if pg_num:
                if len( wpsl_genre[ g ] ) > pg_num:
                    wpsl_genre[ g ] = wpsl_genre[ g ][0:pg_num]
                    vectl_genre[ g ] = vectl_genre[ g ][0:pg_num]
    

#Unit Test
if __name__ == "__main__":
    
    d1 = {'jim':1, 'one':1, 'two':2}
    d2 = {'jim':1, 'one':2, 'two':3, 'three': 9}
    
    print d1
    print d2, "\n"
    
    l = [d1, d2]
    d_all =  VHTools.merge_dicts( *l )
    l_all = VHTools.tf_d_gen( l )
    
    print "all ", d_all
    print "List of Dict res", l_all
    
    print VHTools.term_dict_idxing( d_all )
    
    print VHTools.keep_most_terms( d_all, 3 )
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    