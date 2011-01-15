"""
"""
import re
import os
import codecs
import lxml.etree 
import lxml.html
from lxml.html.clean import Cleaner
import unicodedata
import vectorhandlingtools as vht
import sys
sys.path.append('../../synergeticprocessing/src')
import synergeticpool as sp_mod
import multiprocessing
#import xml.etree.ElementTree as bxml

#Define a local Synergetic Pool with One process for each available CPU
cpu_num = multiprocessing.cpu_count()
#spool = sp_mod.SynergeticPool( local_workers=cpu_num, syn_listener_port=51000 )


class VectGen(object): 

    def __init__(self):
        #For appending the URL of the xtree to the list of Web Pages that the TF (or other) vector has been Generated 
        self.webpg_l = list()
        #For appending the Term Vector of the xtree to the list of Web Pages' Vectors
        self.webpg_vect_l = list()
        #For appending the Ngram Vector of the xtree ot the list of Web Pages' Vectors
        self.ngram_vect_l = list()
        #Defin XPath objects to extract (X)HTML attributes
        ##Extract Text
        self.extract_txt = lxml.etree.XPath("/html/body//text()")
        #Define Regular Expression to extract textual attributes
        ##Whitepace characters [<space>\t\n\r\f\v] matching, for splitting the raw text to terms
        self.white_spliter = re.compile(r'\s+')
        ##Find URL String. Probably anchor text
        self.url_str = re.compile(r'(((ftp://|FTP://|http://|HTTP://|https://|HTTPS://)?(www|[^\s()<>.?]+))?([.]?[^\s()<>.?]+)+?(?=.org|.edu|.tv|.com|.gr|.gov|.uk)(.org|.edu|.tv|.com|.gr|.gov|.uk){1}([/]\S+)*[/]?)')
        #######################The above regular expression needs some more work for not incorrectly catching e.g. ".gr/dd/fdfd" or just ".com"
        ##Comma decomposer
        self.comma_decomp = re.compile(r'[^,]+|[,]+')
        self.comma_str = re.compile(r'[,]+')
        ##Find dot or sequence of dots
        self.dot_str = re.compile(r'[.]+')
        self.dot_decomp = re.compile(r'[^.]+|[.]+')
        ##Symbol term decomposer 
        self.fredsb_clean = re.compile(r'^[^\w]+|[^\w%]+$', re.U) #front-end-symbol-cleaning => fredsb_clean
        ##Find proper number
        self.proper_num = re.compile(r'(^[0-9]+$)|(^[0-9]+[,][0-9]+$)|(^[0-9]+[.][0-9]+$)|(^[0-9]{1,3}(?:[.][0-9]{3})+[,][0-9]+$)|(^[0-9]{1,3}(?:[,][0-9]{3})+[.][0-9]+$)')
        
    def ngrams_vects_from_file(self, args):#file, genre, base_filepath, ng_size=3):
        file, genre, base_filepath, ng_size = args
        for base in base_filepath:
            filepath = base + genre  
            xhtmlfiles_l = [files for path, dirs, files in os.walk(filepath)]
            if xhtmlfiles_l:
                break
        file_ng = ( (filepath + "/"  + file), ng_size )  
        webpg, ngram_vect = self.gen_ngram_vect_re( file_ng )           
        ### 
        print' Returning ', genre, webpg
        return (genre, webpg, ngram_vect)
        
    def ngrams_vects_from_path(self, genre, base_filepath, ng_size=3, multiproc=True):
        #spool = sp_mod.SynergeticPool( local_workers=cpu_num, syn_listener_port=51000 )
        for base in base_filepath:
            filepath = base + genre  
            xhtmlfiles_l = [files for path, dirs, files in os.walk(filepath)]
            if xhtmlfiles_l:
                xhtmlfiles_l = xhtmlfiles_l[0]
                break
        f_ng_sets = [ ( (filepath + "/"  + file), ng_size ) for file in xhtmlfiles_l ] 
        if multiproc:
            print "SPOOL MAP"
            #vect_ll = spool.map(self.gen_ngram_vect, f_ng_sets)
        else:
            print "MAP MAP"
            print filepath
            vect_ll = map(self.gen_ngram_vect, f_ng_sets)
        ###
        for vect_l in vect_ll:
            if vect_l == [None, None, None]:
                vect_ll.remove(vect_l)
        ###
        webpg_l = [ i for i, j in vect_ll] 
        ngram_vect_l = [ j for i, j in vect_ll]
        ###
        idx_l = list()
        for i, wp_vect in enumerate(ngram_vect_l):
            if not wp_vect:
                idx_l.append( i )
        c = -1
        for i in idx_l:
            c += 1
            del ngram_vect_l[ (i - c) ]
            del webpg_l[ (i - c) ]
        ### 
        print(len(webpg_l), len(ngram_vect_l))
        ###
        global_ngram_dict = vht.gterm_d_gen(ngram_vect_l)
        ###
        #spool.join_all() 
        ### 
        return (genre, global_ngram_dict, webpg_l, ngram_vect_l)
    
    def term_vects_from_path(self, genre, base_filepath, multiproc=True):
        for base in base_filepath:
            filepath = base + genre  
            xhtmlfiles_l = [files for path, dirs, files in os.walk(filepath)]
            if xhtmlfiles_l:
                xhtmlfiles_l = xhtmlfiles_l[0] 
        if multiproc:
            pass
            #vect_ll = spool.map(self.gen_term_vect, xhtmlfiles_l, cpu_num)
        else:
            vect_ll = map(self.gen_term_vect, xhtmlfiles_l, cpu_num)
        ###
        for vect_l in vect_ll:
            if vect_l == [None, None]:
                vect_ll.remove(vect_l)
        ###
        webpg_l = [ i for i, j, k in vect_ll] 
        webpg_vect_l = [ k for i, j, k in vect_ll]
        ###
        idx_l = list()
        for i, wp_vect in enumerate(webpg_vect_l):
            if not wp_vect:
                idx_l.append( i )
        c = -1
        for i in idx_l:
            c += 1
            del webpg_vect_l[ (i - c) ]
            del webpg_l[ (i - c) ]
        ### 
        print(len(webpg_l), len(webpg_vect_l))
        ###
        global_term_dict = vht.gterm_d_gen(webpg_vect_l)
        ###
        #spool.join_all()
        ### 
        return (genre, global_term_dict, webpg_l, webpg_vect_l)    
    
    def gen_ngram_vect_re(self, args):
        xhtml, ngram_size = args
        print xhtml
        ##Export Ngrams => 3Grams
        reg_ng_size = r'.{' + str(ngram_size) + '}'
        ngrams = re.compile( reg_ng_size )
        html_tags = re.compile(r'<[^>]+>')
        html_scripts = re.compile(r'<script[^>]*>[\S\s]*</script>')
        proper_html = re.compile(r'<html[^>]*>[\S\s]+</html>')
        try:
            fcod = codecs.open(xhtml, 'r', 'utf8', 'ignore')
        except Exception as e:
            print("faild to load %s: %s", (xhtml, e))
        else:
            try:    
                xhtml_text = fcod.read()
            except Exception as e:
                print("faild to read %s: %s", (xhtml, e))
            finally:
                fcod.close()
        #Check if it is a proper (X)HTML file
        properhtml = proper_html.findall( xhtml_text )
        if not properhtml:
            return [None, None]
        else:
            xhtml_text = properhtml[0]
        #Clean-up Scripts
        xhtml_text_l = html_scripts.split( xhtml_text )
        xt = list()
        for html_chk in xhtml_text_l:
            xt.extend( html_tags.split(html_chk) )
        xhtml_text_l = xt
        #Normalise Unicode String for consistency in text attributes extraction among all corpus' web pages 
        for i in range(len(xhtml_text_l)):
            try:
                xhtml_text_l[i] = unicodedata.normalize('NFKC', xhtml_text_l[i].decode())
            except:
                xhtml_text_l[i] = unicodedata.normalize('NFKC', xhtml_text_l[i])
        #Create the Ngramms Frequency Vectors
        xhtml_NgF = dict()
        #Define the term list that will be used for putting the Terms before we start counting them
        terms_l = list()
        for text_line in xhtml_text_l:
            #Initially split the text to terms separated by whitespaces [ \t\n\r\f\v] 
            terms_l.append( text_line ) #self.white_spliter.split(text_line) )
        #Find and Count NGrams
        for term in terms_l:
            for i in range(len(term)):
                Ngrms_l = ngrams.findall(term[i:])
                for tri_g in Ngrms_l:
                    if tri_g in xhtml_NgF: #if the dictionary of terms has the 'terms' as a key 
                        xhtml_NgF[tri_g] += 1
                    elif tri_g: #None empty strings are accepted 
                        xhtml_NgF[tri_g] = 1
        print "returning ", xhtml 
        return [ xhtml.split('/')[-1], xhtml_NgF ]
    
    def gen_ngram_vect(self, args):
        xhtml, ngram_size = args
        ##Export Ngrams => 3Grams
        reg_ng_size = r'.{' + str(ngram_size) + '}'
        self.ngrams = re.compile( reg_ng_size )
        print xhtml
        xhtml_t = lxml.html.parse( codecs.open(xhtml, 'r', 'utf8', 'ignore') ) 
        if xhtml_t.getroot() == None:
            print "NONE"
            return [None, None]
        #print xhtml_t
        #xcharset = xhtml_d['charset']
        #if not xcharset:
        #xcharset = "utf8"
        #Get the Text of the XHTML in a list of document lines
        try: 
            xhtml_text_l = xhtml_t.xpath("/html/body//text()") #self.extract_txt(xhtml_t)  
        except:
            print "None"
            return [None, None]
        #Normalise Unicode String for consistency in text attributes extraction among all corpus' web pages 
        for i in range(len(xhtml_text_l)):
            try:
                xhtml_text_l[i] = unicodedata.normalize('NFKC', xhtml_text_l[i].decode())
            except:
                xhtml_text_l[i] = unicodedata.normalize('NFKC', xhtml_text_l[i])
        #Create the Ngramms Frequency Vectors
        xhtml_NgF = dict()
        #Define the term list that will be used for putting the Terms before we start counting them
        terms_l = list()
        for text_line in xhtml_text_l:
            #Initially split the text to terms separated by whitespaces [ \t\n\r\f\v] 
            terms_l.extend( self.white_spliter.split(text_line) )
        #Find and Count NGrams
        for term in terms_l:
            for i in range(len(term)):
                Ngrms_l = self.ngrams.findall(term[i:])
                for tri_g in Ngrms_l:
                    if tri_g in xhtml_NgF: #if the dictionary of terms has the 'terms' as a key 
                        xhtml_NgF[tri_g] += 1
                    elif tri_g: #None empty strings are accepted 
                        xhtml_NgF[tri_g] = 1
        del xhtml_t ### I DONT THINK THAT THIS IS THE SOLUTION FOR PREVENTING MEMORY LEAKAGE
        print "returning ", xhtml 
        return [ xhtml.split('/')[-1], xhtml_NgF ]
        
    def gen_term_vect(self, xhtml):
        xhtml_t = lxml.html.parse( open(xhtml, "r") ) 
        print "etree"
        if xhtml_t.getroot() == None:
            print "NONE"
            return [None, None]
        #print xhtml_t
        #xcharset = xhtml_d['charset']
        #if not xcharset:
        xcharset = "utf8"
        #Get the Text of the XHTML in a list of document lines
        try: 
            xhtml_text_l = self.extract_txt(xhtml_t)
        except:
            print "None"
            return [None, None]
        #Normalise Unicode String for consistency in text attributes extraction among all corpus' web pages 
        for i in range(len(xhtml_text_l)):
            xhtml_text_l[i] = unicodedata.normalize('NFKC', xhtml_text_l[i].decode()) 
        #Create the Word Term Frequency Vectors 
        xhtml_TF = dict()
        #Define the term list that will be used for putting the Terms before we start counting them
        terms_l = list()
        for text_line in xhtml_text_l:
            #Initially split the text to terms separated by whitespaces [ \t\n\r\f\v] 
            terms_l.extend( self.white_spliter.split(text_line) )
        #Count and remove the Numbers form the terms_l
        terms_l = self.get_proper_numbers(terms_l, xhtml_TF)
        #Decompose the terms to sub-terms of any symbol but comma (,) and comma sub-term(s)
        terms_l = self.get_comma_n_trms(terms_l, xhtml_TF)
        #Split term to words upon dot (.) and dot needs special treatment because we have the case of . or ... and so on
        terms_l = self.get_dot_n_trms(self, terms_l, xhtml_TF)
        #Count and clean-up the non-alphanumeric symbols ONLY from the Beginning and the End of the terms
        ##except dot (.) and percentage % at the end for the term)
        terms_l = self.get_propr_trms_n_symbs(terms_l, xhtml_TF)
        #Finally! count the term frequencies (with any Noise unfortunately remains)    
        for term in terms_l:            
            if term in xhtml_TF: #if the dictionary of terms has the 'terms' as a key 
                xhtml_TF[term] += 1
            elif term:
                xhtml_TF[term] = 1                    
        #Append the URL of the xtree to the list of Web Pages that the TF (or other) vector has been Generated 
        #self.webpg_l.append( xhtml_d['filename'] )
        print "etree : Filters Done!"
        del xhtml_t ### I DONT THINK THAT THIS IS THE SOLUTION FOR PREVENTING MEMORY LEAKAGE
        return [ xhtml.split('/')[-1], xhtml_TF] 
    
    def get_proper_numbers(self, terms_l, xhtml_TF):
        num_free_tl = list()
        for term in terms_l:
            num_term_l = self.proper_num.findall(term)
            if num_term_l: #if a number found the the term should be the number so we keep it as it is
                #for i in range(len(num_term_l[0])):
                #    use this for-loop in case you want to know the exact form of the number.
                #    Each from has a position with the following order 1)xxxxxx 2)xxxx,xxxx 3)xxxx.xxxx 4)333.333.333...333,xxxxxx 5)333,333,333,333,,,333.xxxxxx
                if term in xhtml_TF: #if the dictionary of terms has the 'terms' as a key 
                    xhtml_TF[term] += 1
                else:    
                    xhtml_TF[term] = 1
            else:
                #Keep only Non-Number terms or Non-proper-numbers
                num_free_tl.append(term)
        return num_free_tl
    
    def get_comma_n_trms(self, terms_l, xhtml_TF):
        comma_free_tl = list()
        for term in terms_l:
            #Decompose the terms that in their char set include comma symbol to a list of comma separated terms and the comma(s) 
            decomp_term_l = self.comma_decomp.findall(term)
            if len(decomp_term_l) > 1:
                for subterm in decomp_term_l:
                    if self.comma_str.findall(subterm):
                        if subterm in xhtml_TF: #if the dictionary of terms has the 'terms' as a key 
                            xhtml_TF[subterm] += 1
                        else:    
                            xhtml_TF[subterm] = 1
                    else: #if the substring is not a comma string then forward for farther analysis 
                        comma_free_tl.append(subterm)
            else:
                #Keep only the terms that are already free of commas because the other have been already decomposed and counted
                comma_free_tl.append(term)
        #use the comma_free terms list as the terms list to continue processing
        return comma_free_tl
    
    def get_dot_n_trms(self, terms_l, xhtml_TF):
        dot_free_tl = list()
        for term in terms_l:
            decomp_term = self.dot_decomp.findall(term)
            dec_trm_len = len(decomp_term)
            if dec_trm_len > 1 and dec_trm_len <= 3: 
                #Here we have the cases of ...CCC or .CC or CC.... or CCC. or CC.CCC or CCCC....CCCC so keep each sub-term
                for sub_term in decomp_term:
                    if self.dot_str.findall(sub_term):
                        if sub_term in xhtml_TF: 
                            xhtml_TF[sub_term] += 1
                        else:
                            xhtml_TF[sub_term] = 1
                    else: #give the new terms for farther analysis
                        dot_free_tl.append(sub_term)
            elif dec_trm_len > 3: #i.e. Greater thatn 3
                #Remove the first and the last dot sub-string and let the rest of the term as it was (but the prefix and suffix of dot(s))
                sub_term_l = list()
                #Extract dot-sequence prefix if any 
                if self.dot_str.findall(decomp_term[0]):
                    sub_term_l.append( decomp_term.pop(0) )
                #Extract dot-sequence suffix if any
                l_end = len(decomp_term) - 1
                if self.dot_str.findall(decomp_term[l_end]):
                    sub_term_l.append( decomp_term.pop(l_end) )
                #Count dot-sequence terms 
                for sub_term in sub_term_l:                  
                    if sub_term in xhtml_TF: 
                        xhtml_TF[sub_term] += 1
                    else:
                        xhtml_TF[sub_term] = 1
                #Re-compose the term without suffix/prefix dot-sequence and give it for further analysis 
                dot_free_tl.append( "".join(decomp_term) )
            else:
                if self.dot_str.findall(term): #in case of one element in the list check if it is a dot-sequence
                    if term in xhtml_TF: 
                        xhtml_TF[term] += 1
                    else:
                        xhtml_TF[term] = 1
                else: #keep already the dot-free terms    
                    dot_free_tl.append(term) 
        return  dot_free_tl
        
    def get_propr_trms_n_symbs(self, terms_l, xhtml_TF):
        clean_term_tl = list()
        for term in terms_l:
            #Get the 
            symb_term_l = self.fredsb_clean.findall(term)
            if symb_term_l:
                #Keep and count the symbols found 
                for symb in symb_term_l:
                    if symb in xhtml_TF: #if the dictionary of terms has the 'terms' as a key 
                        xhtml_TF[symb] += 1
                    else: 
                        xhtml_TF[symb] = 1
                clean_trm = self.fredsb_clean.sub('', term)
                if clean_trm in xhtml_TF: #if the dictionary of terms has the 'terms' as a key 
                    xhtml_TF[clean_trm] += 1
                elif clean_trm: #if not empty string (Just in case)
                    xhtml_TF[clean_trm] = 1
            else:
                #Keep only the terms that are already free of commas because the other have been already decomposed and counted
                clean_term_tl.append(term)
        #use the comma_free terms list as the terms list to continue processing
        terms_l = clean_term_tl


