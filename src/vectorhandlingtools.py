"""

"""

import eventlet
import codecs
import os
import sys
from basicfiletools import BaseFileTools
#from scgenrelerner_svmbased import *


class VHTools(BaseFileTools):
    
    @staticmethod
    def keep_atleast(terms_d, terms_amount):
        """ keep_most(): is getting a dictionary of Terms-Frequencies and 
            the amount of Terms to return as arguments. It is returning the number
            of Terms equal to the argument 'terms_amount' with the Highest Frequency.
            However if the subsequent terms have the same Frequency with the last
            one if the Returned dictionary then it will include this terms. """
        terms_l = [(v, k) for (k, v) in terms_d.iteritems()]
        terms_l.sort()
        terms_l.reverse()
        atlest_terms_l = terms_l[0:terms_amount]
        print len(atlest_terms_l) 
        last_freq = atlest_terms_l[-1][0]
        #print last_freq
        for freq, term in terms_l[terms_amount:]:
            if freq == last_freq:
                atlest_terms_l.append( (freq, term) )
        terms_d = dict( [ (k, v) for (v, k) in atlest_terms_l ] )
        return terms_d

    @staticmethod
    def keep_most(terms_d, terms_amout):
        """ keep_most(): is getting a dictionary of Terms-Frequencies and 
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
    def __load_tf_dict(filename, encoding='utf-8', error_handling='strict', force_lower_case=False):
        """ __load_tf_dict(): do not use this function prefer the VHTools.load_tf_dict(). 
            This function is getting a filename and a lower case force option and returns a 
            Term-Frequency dictionary loaded from the file given as argument. """
        try:
            fenc = codecs.open( filename, 'rb', encoding, error_handling)
        except IOError as e:
            print("VHTools.load_dict() FILE %s ERROR: %s" % (filename,e))
            return None
        #The following for loop is an alternative approach to reading lines instead of using f.readline() or f.readlines()
        tf_d = dict()
        try:
            wholefile = fenc.read()
            for fileline in wholefile.split("\n~\n~\n")[:-1]:
                line = fileline #.rstrip()
                if len(line.split(" ~~> ")) != 2:
                    print line
                Term, Freq = tuple( line.split(" ~~> ") )
                if force_lower_case: 
                    tf_d[ Term.lower() ] = float( Freq )
                else:
                    tf_d[ Term ] = float( Freq )
        except Exception as e:
            print("VHTools.__load_tf_dict() Error: %s" % e)
            return None
        finally:
            fenc.close()    
        return tf_d  
    
    @staticmethod
    def load_tf_dict(filename_l, encoding='utf-8', error_handling='strict', force_lower_case=False):
        """ load_tf_dict(): is getting a filename or a (filename list) and lower case force option
            as arguments. It returns a Term-Frequency Dictionary which is a merged dictionary of all
            TF dictionaries given a argument """
        if isinstance(filename_l, str):
            return VHTools.__load_tf_dict(filename_l, encoding, error_handling, force_lower_case)
        elif isinstance(filename_l, list):
            mrgd_tf_d = dict()
            for filename in filename_l:
                tf_d = VHTools.__load_tf_dict(filename, encoding, error_handling, force_lower_case)
                for Term, Freq in tf_d.items():
                    if Term in mrgd_tf_d: 
                        mrgd_tf_d[ Term ] += float(Freq) 
                    else:
                        mrgd_tf_d[ Term ] = float(Freq)
            return mrgd_tf_d
        else:
            raise Exception("A String or a list of Strings was Expected as input")
    
    @staticmethod
    def load_tf_dict_NUM(filename_l, encoding='utf-8', error_handling='strict', force_lower_case=False):
        """ load_tf_dict(): is getting a filename or a (filename list) and lower case force option
            as arguments. It returns a Term-Frequency Dictionary which is a merged dictionary of all
            TF dictionaries given a argument """
        if isinstance(filename_l, str):
            return VHTools.__load_tf_dict(filename_l, encoding, error_handling, force_lower_case)
        elif isinstance(filename_l, list):
            mrgd_tf_d = dict()
            for filename in filename_l:
                tf_d = VHTools.__load_tf_dict(filename, encoding, error_handling, force_lower_case)
                for Term, Freq in tf_d.items():
                    if Term in mrgd_tf_d: 
                        mrgd_tf_d[ int(Term) ] += float(Freq) 
                    else:
                        mrgd_tf_d[ int(Term) ] = float(Freq)
            return mrgd_tf_d
        else:
            raise Exception("A String or a list of Strings was Expected as input")
    
    
    @staticmethod
    def load_tfd_frmpaths(basepath, filepath_l, encoding, error_handling, force_lower_case=False):
        """ laod_tf_frmpaths: is getting a list of paths as argument and a base path as optional argument. 
            Returns a merge of all Term-Frequency Dictionaries found in the file paths list. """
        fname_lst = VHTools.file_list_frmpaths(basepath, filepath_l)
        return VHTools.load_tf_dict(fname_lst, encoding, error_handling, force_lower_case)

    @staticmethod
    def __load_tf_dict_l(filename, line_lim=0 , encoding='utf-8', error_handling='strict', force_lower_case=False):
        """ __load_tf_dict_l(): is getting a filename as argument and force lower_case as optional argument.
            It returns an a list for TF dictionaries and a list of the web-pages where the TF Dictionaries are 
            related to. If line_lim is set above 0 then If a file contains more than one line (i.e. web page vectors) keep 
            only the amount of pages requested in this argument  """
        try:
            fenc = codecs.open( str(filename), 'rb', 'utf-8', 'strict')
        except IOError, e:
            print("VHTools.__load_dict_l() FILE %s ERROR: %s" % (filename,e))
            return None
        #The following for loop is an alternative approach to reading lines instead of using f.readline() or f.readlines()
        wps_l = list()
        vect_l = list()
        try:
            lines_cnt = 0
            wholetxt = fenc.read()
            for fileline in wholetxt.split("\n~\n~\n")[:-1]:
                line = fileline #.rstrip()
                wp_name, wp_tf_d = tuple( line.split(" ~~> ") ) #BE CAREFULL with SPACES
                wps_l.append( wp_name )
                composed_terms = wp_tf_d.split('\t~,~\t')
                vect_dict = dict()   
                for comp_term in composed_terms:        
                    Term, Freq = tuple( comp_term.split(' ~:~ ') )
                    if force_lower_case:
                        vect_dict[ Term.lower() ] = float( Freq )
                    else:    
                        vect_dict[ float(Term) ] = float( Freq )
                vect_l.append( vect_dict )
                #If a file contains more than one line (i.e. web page vectors) keep 
                #only the amount of pages requested in the line_lim argument
                lines_cnt += 1
                if lines_cnt == line_lim:
                    break
        except Exception as e:
            print("VHTools.__load_dict_l() FILE %s ERROR: %s" % (filename,e))
            return None
        finally:
            fenc.close()      
        return (wps_l, vect_l)
    
    @staticmethod
    def load_tf_dict_l(filename_l, page_lim=0, encoding='utf-8', error_handling='strict', force_lower_case=False):
        """ loadtf_dict_l(): is getting a filename or a filename list as first arguments and
            a lower case force option. If page_lim is set above 0 then If a file contains more than one line (i.e. web page vectors) 
            or a folder contains more than one files then keeps only the amount of pages requested in this argument either
            because the amount of lines in a file have reached the argument or because the amount of file have been loaded have. """
        if isinstance(filename_l, str):
            return VHTools.__load_tf_dict_l(filename_l, page_lim, encoding, error_handling, force_lower_case)
        elif isinstance(filename_l, list):
            mrgd_wps_l = list()
            mrgd_vect_l = list()
            for filename in filename_l:
                wps_l, vect_l = VHTools.__load_tf_dict_l(filename, page_lim, encoding, error_handling, force_lower_case)
                mrgd_wps_l.extend( wps_l )
                mrgd_vect_l.extend( vect_l )
                #If a the merged list of web pages reached the the amount of pages requested 
                #in the page_lim argument the loop stops
                if len(mrgd_wps_l) == page_lim: 
                    break
            return (mrgd_wps_l, mrgd_vect_l)
        else:
            raise Exception("A String or a list of Strings was Expected as input")
        
    @staticmethod
    def load_tfd_l_frmpaths(basepath, filepath_l, page_lim=0, encoding='utf-8', error_handling='strict', force_lower_case=False):
        """ load_tfd_l(): is getting a list of file-paths, a base-path, and a lower case force option
            as arguments. In addition it has the page_lim argument for constraining the amount of web page vectors to be
            loaded if requested using this argument. It returns a list of TF-Dictionaries and a list of the Web-pages related to 
            the TF-Dictionaries, of all the files found in the file-paths lists."""
        fname_lst = VHTools.file_list_frmpaths(basepath, filepath_l)
        return VHTools.load_tf_dict_l(fname_lst, page_lim, encoding, error_handling, force_lower_case)
            
    @staticmethod
    def __tf2idxf(tf_d, tidx_d):
        """ __tf2idxf(): Don't use it directly, use tf2idxf instead.
            This function is getting a TF dictionary representing the TF Vector,
            and a TF-Index as defined in VHTools.tf_dict_idxing(). It returns
            a Index-Frequency dictionary where each term of the TF dictionary has been 
            replaced with the Index number of the TF-Index. In case the term of the 
            TF Dictionary is not in the TF-Index then the term is just Dropped. Therefore,
            the Index-Frequency dictionary it will no more include the missing (from TF-Index) term. """
        idxed_d = dict() 
        for term, freq in tf_d.items():
            if term in tidx_d:
                idxed_d[ tidx_d[term] ] = freq
            #else: DROP THE TERM
        return idxed_d
    
    @staticmethod
    def tf2idxf(tf_d_l, tf_idx_d):
        """ tf2idxf(): is getting a TF-Dictionary or a list of TF-Dictionaries and TF-Index. It applies
            the VHTools.__tf2idxf() function to the dictionaries and returns a list or single TF-Dictionary
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
    def save_tf_dct(filename, records, encoding='utf-8', error_handling='strict'):
        """ save_tf_dct(): is getting a filename string and a TF-Dictionary saves 
            the dictionary to a file with utf-8 Encoding. """
        try:
            #Codecs module is needed to assure the proper saving of string in UTF-8 encoding. 
            fenc = codecs.open( filename, 'wb', encoding, error_handling)
        except IOError:
            return None 
        try:
            for rec in records:  
                fenc.write( str(rec) + " ~~> "  + str(records[rec]) + "\n~\n~\n" ) # Write a string to a file 
        except Exception as e:
            print("ERROR WRITTING FILE: %s -- %s" % (filename, e))
        finally:
            fenc.close()
        return True           

    @staticmethod
    def save_tf_dct_lst(filename, records, index, encoding='utf-8', error_handling='strict'):
        """ save_tf_dct_lst(): is getting a filename string a list of TF-Dictionaries and a List of Web-Pages
            related to the TF-Dictionaries and saves them to a file in the form <webpage-filename> ~~> <TF-Dictionary> """
        try:
            #Codecs module is needed to assure the proper saving of string in UTF-8 encoding.
            fenc = codecs.open( filename, 'wb', encoding, error_handling) 
        except IOError as e:
            print "save_dct_lst Error: ", e
            return None 
        try: 
            idx_len = len(index)
            for i in range(idx_len):
                fenc.write(index[i] + " ~~> ")
                rcds_len = len(records[i])
                for rec_no, rec in enumerate(records[i]):
                    fenc.write( str(rec) + " ~:~ "  + str(records[i][rec]))
                    if rec_no != rcds_len - 1:
                        fenc.write("\t~,~\t")
                    else:
                        print 'NO LAST DILIMETER' 
                fenc.write("\n~\n~\n") 
        except Exception as e:
            print("ERROR WRITTING FILE: %s -- %s ---- %s" % (filename, e, rec))
        finally:
            fenc.close()
        return True       
    
class GreenVHTools(VHTools):
    """ GreenVHTools Class is a GreenLet/Eventlet version of VHTools Class.
        Actually it just overrides the load_tf_dict() and load_tf_dict_l() with an Eventlet-Based
        version of them """
        
    @staticmethod
    def load_tf_dict(filename_l, encoding='utf-8', error_handling='strict', force_lower_case=False):
        """ """
        if isinstance(filename_l, str):
            return VHTools.__load_dict_l(filename_l, encoding, error_handling, force_lower_case)
        elif isinstance(filename_l, list):
            mrgd_wps_l = list()
            mrgd_vect_l = list()
            gpool = eventlet.GreenPool(1000)
            force_lower= map( lambda x: force_lower_case, range(len(filename_l)) )
            enc  = map( lambda x:  encoding, range(len(filename_l)) )
            err_handling = map( lambda x: error_handling, range(len(filename_l)) )
            for wps_l, vect_l in gpool.imap(GreenVHTools.__load_dict_l, filename_l, enc,  err_handling, force_lower):
                mrgd_wps_l.extend( wps_l )
                mrgd_vect_l.extend( vect_l )
            return (mrgd_wps_l, mrgd_vect_l)
        else:
            raise Exception("A String or a list of Strings was Expected as input")
    
    @staticmethod
    def load_tf_dict_l(filename_l, encoding='utf-8', error_handling='strict', force_lower_case=False):
        if isinstance(filename_l, str):
            return GreenVHTools.__load_tf_dict(filename_l, encoding, error_handling, force_lower_case)
        elif isinstance(filename_l, list):
            gpool = eventlet.GreenPool(1000)
            force_lower= map( lambda x: force_lower_case, range(len(filename_l)) )
            enc  = map( lambda x:  encoding, range(len(filename_l)) )
            err_handling = map( lambda x: error_handling, range(len(filename_l)) )
            mrgd_tf_d = GreenVHTools.__load_tf_dict(filename_l[0], encoding, error_handling, force_lower_case)
            for tf_d in gpool.imap(GreenVHTools.__load_tf_dict, filename_l[1:], enc,  err_handling, force_lower):
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
    d3 = {'jim':1, 'one':2, 'two':3, 'three': 9, 'Five': 99}
    
    print d1
    print d2, "\n"
    
    l = [d1, d2]
    d_all =  VHTools.merge_tf_dicts( *l )
    l_all = VHTools.gen_tfd_frmlist( l )
    
    print "all ", d_all
    print "List of Dict res", l_all
    
    tidx_d = VHTools.tf2tidx( d_all )
    print tidx_d
    idxf =  VHTools.tf2idxf(d3, tidx_d)
    print idxf
    #from trainevaloneclssvm import TermVectorFormat as vf
    #print vf.tf2bin([idxf], tidx_d, 3)
    
    print VHTools.keep_most( d_all, 3 )
    
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
    
    
    