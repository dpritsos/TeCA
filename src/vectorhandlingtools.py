"""

"""

import eventlet
import codecs
import os
#from scgenrelerner_svmbased import *


class Depricated_Functions(object):
    
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



class VHTools(object):
    
    @staticmethod
    def __exec_on_files_frmpaths(func, basepath, filepath_l, force_lower_case=False):
        """ __exec_on_files_frmpaths(): is executing a function given as the first argument to all the
            files found in the file-paths list given third and second argument. Optionally a fore lower
            case argument can be given. """
        if basepath is None:
            basepath = '' 
        if isinstance(filepath_l, str):
            flist = [ files_n_paths[2] for files_n_paths in os.walk(filepath_l) ]
            flist = flist[0]
            fname_lst = [ str(basepath) + filepath_l + fname for fname in flist ]
        elif isinstance(filepath_l, list):
            fname_lst = list()
            for filepath in filepath_l:
                flist = [ files_n_paths[2] for files_n_paths in os.walk(filepath) ]
                flist = flist[0]
                fname_lst.extend( [ str(basepath) + filepath + fname for fname in flist ] )
        else:
            raise Exception("A String or a list of Strings was Expected as input - Stings should be file-paths")
        return func(fname_lst, force_lower_case)
    
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
        """ mege_tf_dicts(): is getting a set of term-frequency dictionaries as list of
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
    def gen_tfd_frmlist(tf_d_l):
        """ gen_tfd_frmlist(): is getting a list of Term-Frequency Dictionaries and creates a 
            TF Dictionary of all terms occurred in the list. """
        return VHTools.merge_tf_dicts( *tf_d_l )
    
    @staticmethod
    def tf2tidx(term_d):
        """ tf2tidx(): is getting a Term-Frequency dictionary and returns one
            with terms-index number. The index number is just their position in the
            descending order sorted list of dictionary keys. """
        term_l = term_d.keys()
        term_l.sort()
        idx = range( len(term_l) + 1 )
        term_idx_d = dict( zip( term_l , idx[1:] ) )
        return term_idx_d
    
    @staticmethod
    def __load_tf_dict(filename, force_lower_case=False):
        """ __load_tf_dict(): do not use this function prefer the VHTools.load_tf_dict(). 
            This function is getting a filename and a lower case force option and returns a 
            Term-Frequency dictionary loaded from the file given as argument. """
        try:
            f = codecs.open( filename, 'r', 'utf-8', 'strict')
        except IOError as e:
            print("VHTools.load_dict() FILE %s ERROR: %s" % (filename,e))
            return None
        #The following for loop is an alternative approach to reading lines instead of using f.readline() or f.readlines()
        tf_d = dict()
        try:
            for fileline in f:
                line = fileline.replace('\n', '')
                Term, Freq = tuple( line.split(" => ") )
                if force_lower_case: 
                    tf_d[ Term.lower() ] = float( Freq )
                else:
                    tf_d[ Term ] = float( Freq )
        except Exception as e:
            print("VHTools.__load_tf_dict() Error: %s" % e)
            return None
        finally:
            f.close()    
        return tf_d  
    
    @staticmethod
    def load_tf_dict(filename_l, force_lower_case=False):
        """ load_tf_dict(): is getting a filename or a (filename list) and lower case force option
            as arguments. It returns a Term-Frequency Dictionary which is a merged dictionary of all
            TF dictionaries given a argument """
        if isinstance(filename_l, str):
            return VHTools.__load_tf_dict(filename_l, force_lower_case)
        elif isinstance(filename_l, list):
            mrgd_tf_d = dict()
            for filename in filename_l:
                tf_d = VHTools.__load_tf_dict(filename, force_lower_case)
                for Term, Freq in tf_d.items():
                    if Term in mrgd_tf_d: 
                        mrgd_tf_d[ Term ] += Freq 
                    else:
                        mrgd_tf_d[ Term ] = Freq
            return mrgd_tf_d
        else:
            raise Exception("A String or a list of Strings was Expected as input")
    
    @staticmethod
    def load_tfd_frmpaths(basepath, filepath_l, force_lower_case=False):
        """ laod_tf_frmpaths: is getting a list of paths as argument and a base path as optional argument. 
            Returns a merge of all Term-Frequency Dictionaries found in the file paths list. """
        return VHTools.__exec_on_files_frmpaths( VHTools.load_tf_dict, basepath, filepath_l, force_lower_case=False )

    @staticmethod
    def __load_tf_dict_l(filename, force_lower_case=False):
        """ __load_tf_dict_l(): is getting a filename as argument and force lower_case as optional argument.
            It returns an a list for TF dictionaries and a list of the web-pages where the TF Dictionaries are 
            related to. """
        try:
            f = codecs.open( str(filename), "r", "utf-8")
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
                wp_tf_d.replace('\n', '')
                composed_terms = wp_tf_d.split('\t')
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
            print("VHTools.__load_tf_dict_l() Error: %s" % e)
            return None
        finally:
            f.close()     
        return (wps_l, vect_l)
    
    @staticmethod
    def load_tf_dict_l(filename_l, force_lower_case=False):
        """ loadtf_dict_l(): is getting a filename or a filename list as first arguments and
            a lower case force option. """
        if isinstance(filename_l, str):
            return VHTools.__load_tf_dict_l(filename_l, force_lower_case)
        elif isinstance(filename_l, list):
            mrgd_wps_l = list()
            mrgd_vect_l = list()
            for filename in filename_l:
                wps_l, vect_l = VHTools.__load_tf_dict_l(filename, force_lower_case)
                mrgd_wps_l.extend( wps_l )
                mrgd_vect_l.extend( vect_l )
            return (mrgd_wps_l, mrgd_vect_l)
        else:
            raise Exception("A String or a list of Strings was Expected as input")
        
    @staticmethod
    def load_tfd_l_frmpaths(basepath, filepath_l, force_lower_case=False):
        """ load_tfd_l(): is getting a list of file-paths, a base-path, and a lower case force option
            as arguments. It returns a list of TF-Dictionaries and a list of the Web-pages related to 
            the TF-Dictionaries, of all the files found in the file-paths lists."""
        return VHTools.__exec_on_files_frmpaths( VHTools.load_tf_dict_l, basepath, filepath_l, force_lower_case=False )
            
    @staticmethod
    def __tf2idxf(tf_d, tf_idx_d):
        """ __tf2idxf(): Don't use it directly, use tf2idxf instead.
            This function is getting a TF dictionary representing the TF Vector,
            and a TF-Index as defined in VHTools.tf_dict_idxing(). It returns
            a Index-Frequency dictionary where each term of the TF dictionary has been 
            replaced with the Index number of the TF-Index. """
        idxed_d = dict() 
        for term, freq in tf_d.items():
            idxed_d[ tf_idx_d[term] ] = freq
        return idxed_d
    
    @staticmethod
    def tf2idxf(tf_d_l, tf_idx_d):
        """ tf2idxf(): is getting a TF-Dictionary or a list of TF-Dictionaries and TF-Index. It applys
            the VHTools.__tf2idxf() function to the dictionaries and returns a list or single TF-Dictioary
            depending on the input. """
        if isinstance(tf_d_l, list):
            idxed_d = list()
            for tf_d in tf_d_l:
                idxed_d.append( VHTools.__tf2idxf(tf_d, tf_idx_d) )
            return idxed_d
        elif isinstance(tf_d_l, dict):
            return VHTools.__tf2idxf(tf_d_l, tf_idx_d)
        else:
            raise Exception("Dictionary or a List of Dictionaries was expected as fist input argument")

    @staticmethod
    def save_tf_dct(filename, records):
        """ save_tf_dct(): is getting a filename string and a TF-Dictionary saves 
            the dictionary to a file with utf-8 Encoding. """
        try:
            #Codecs module is needed to assure the proper saving of string in UTF-8 encoding. 
            f = codecs.open( filename, "w", "utf-8") 
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
    def save_tf_dct_lst(filename, records, index):
        """ save_tf_dct_lst(): is getting a filename string a list of TF-Dictionaries and a List of Web-Pages
            related to the TF-Dictionaries and saves them to a file in the form <webpage-filename> => <TF-Dictionary> """
        try:
            #Codecs module is needed to assure the proper saving of string in UTF-8 encoding.
            print "SAVING, ", filename
            f = codecs.open( filename, "w", "utf-8") 
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

    
class GreenVHTools(VHTools):
    """ GreenVHTools Class is a GreenLet/Eventlet version of VHTools Class.
        Actually it just overrides the load_tf_dict() and load_tf_dict_l() with an Eventlet-Based
        version of them """
        
    @staticmethod
    def load_tf_dict(filename_l, force_lower_case=False):
        """ """
        if isinstance(filename_l, str):
            return VHTools.__load_dict_l(filename_l, force_lower_case)
        elif isinstance(filename_l, list):
            mrgd_wps_l = list()
            mrgd_vect_l = list()
            gpool = eventlet.GreenPool(1000)
            force_lower= map( lambda x: force_lower_case, range(len(filename_l)) )
            for wps_l, vect_l in gpool.imap(GreenVHTools.__load_dict_l, filename_l, force_lower):
                mrgd_wps_l.extend( wps_l )
                mrgd_vect_l.extend( vect_l )
            return (mrgd_wps_l, mrgd_vect_l)
        else:
            raise Exception("A String or a list of Strings was Expected as input")
    
    @staticmethod
    def load_tf_dict_l(filename_l, force_lower_case=False):
        if isinstance(filename_l, str):
            return GreenVHTools.__load_tf_dict(filename_l, force_lower_case)
        elif isinstance(filename_l, list):
            gpool = eventlet.GreenPool(1000)
            force_lower= map( lambda x: force_lower_case, range(len(filename_l)) )
            mrgd_tf_d = GreenVHTools.__load_tf_dict(filename_l[0], force_lower)
            for tf_d in gpool.imap(GreenVHTools.__load_tf_dict, filename_l[1:], force_lower):
                for Term, Freq in tf_d.items():
                    if Term in mrgd_tf_d: 
                        mrgd_tf_d[ Term ] += Freq 
                    else:
                        mrgd_tf_d[ Term ] = Freq
            return mrgd_tf_d
        else:
            raise Exception("A String or a list of Strings was Expected as input")
    
    
#Unit Test
if __name__ == "__main__":
    
    d1 = {'jim':1, 'one':1, 'two':2}
    d2 = {'jim':1, 'one':2, 'two':3, 'three': 9}
    
    print d1
    print d2, "\n"
    
    l = [d1, d2]
    d_all =  VHTools.merge_tf_dicts( *l )
    l_all = VHTools.gen_tfd_frmlist( l )
    
    print "all ", d_all
    print "List of Dict res", l_all
    
    print VHTools.tf2tidx( d_all )
    
    print VHTools.keep_most_terms( d_all, 3 )
    
    filename = "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/news/ngrams_corpus_dictionaries/news.ncorpd"
    print "TF Dictionary Loaded: ", len( VHTools.load_tf_dict(filename , True) )
    flist = ["/home/dimitrios/Documents/Synergy-Crawler/saved_pages/news/ngrams_corpus_dictionaries/news.ncorpd", 
             "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/blogs/ngrams_corpus_dictionaries/blogs.ncorpd"]
    print "List of TF Dictionary Loaded: ", len( VHTools.load_tf_dict(flist , True) )
    fpath_list = ["/home/dimitrios/Documents/Synergy-Crawler/saved_pages/news/ngrams_corpus_dictionaries/", 
                  "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/blogs/ngrams_corpus_dictionaries/"] 
    print "Merged TF Dictionary from files of a path or path list", len( VHTools.load_tfd_frmpaths(None, fpath_list , True) )
    tidx_d = VHTools.tf2tidx( VHTools.load_tfd_frmpaths(None, fpath_list , True) )
    print "From Term-Frequency Dictionary to Term-Index Dictionary after Indexing Process: ", tidx_d
    
    filename = "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/news/ngrams_corpus_webpage_vectors/www.latimes.com.49.html.news.ncvect"
    flist = ["/home/dimitrios/Documents/Synergy-Crawler/saved_pages/news/ngrams_corpus_webpage_vectors/www.latimes.com.49.html.news.ncvect",
             "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/news/ngrams_corpus_webpage_vectors/www.latimes.com.50.html.news.ncvect"]
    wpgs, tf_d = VHTools.load_tf_dict_l(filename , True)
    print "TF Dictionary list Loaded: ", len( wpgs ), len( tf_d )
    print "TF Dict of the last loading: ", tf_d
    wpgs_l, tf_d_l = VHTools.load_tf_dict_l(flist, True)
    print "Merged list of TF Dictionary lists Loaded: ", len( wpgs ), len( tf_d_l )
    print "TF Dict of the last loading: ", tf_d_l
    merged_tf_d = VHTools.gen_tfd_frmlist(tf_d_l)
    termidx_d = VHTools.tf2tidx( merged_tf_d )
    print "TF Dict to Term-Index Dict from TF-Dictionary List (length): ", len( termidx_d )
    print "TF Dict List to  Index-Frequency Dict List: ", VHTools.tf2idxf(tf_d, termidx_d)
    fpath_list = ["/home/dimitrios/Documents/Synergy-Crawler/saved_pages/news/ngrams_corpus_webpage_vectors/",
                  "/home/dimitrios/Documents/Synergy-Crawler/saved_pages/blogs/ngrams_corpus_webpage_vectors/"]
    #wpgs, tf_d = VHTools.load_tfd_l_frmpaths(None, fpath_list, True)
    #print "Merged list of TF Dictionary lists Loaded from a list of paths: ", len( wpgs ), len( tf_d )
    
    
    