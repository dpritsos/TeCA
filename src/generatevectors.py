
import os
from vectorhandlingtools import VHTools 
from html2vector import RegexHTML2NgFTools, RegexHTML2TFTools
import codecs

class HTML2TF(VHTools, RegexHTML2TFTools):
    
    def __init__(self):
        RegexHTML2TFTools.__init__(self)
    
    def html2tfv_n_tfd(self, basefrom_l, pathsfrom, tfv_file, tfd_file, err_file, load_encoding='utf-8', save_encoding='utf-8', error_handling='strict', low_mem='True'):
        #Find witch base path is present
        if isinstance(basefrom_l, list):
            for basefrom in basefrom_l:
                filepath = basefrom + pathsfrom    
                flist = [files for path, dirs, files in os.walk(filepath)]
                if flist:
                    break
        else:
            basefrom = basefrom_l
        print filepath 
        wpg_l, tf_d_l = self.tf_d_frmpaths(basefrom, pathsfrom, load_encoding, error_handling, low_mem)
        rmv_wpgs_l = list()
        rmv_idxs_l = list()
        with codecs.open(basefrom + '/' + err_file, 'w', 'utf-8', 'strict') as  fenc:      
            for i in range(len(wpg_l)):
                if not tf_d_l[i]:
                    fenc.write( str(wpg_l[i]) + ' => ' + str(tf_d_l[i]) + '\n' )
                    rmv_wpgs_l.append( wpg_l[i] )
                    rmv_idxs_l.append( i )
        for i, rmv_i in enumerate(rmv_idxs_l):
            tf_d_l.pop( rmv_i - i )
        for wpg in rmv_wpgs_l:
            wpg_l.remove( wpg )
        if self.save_tf_dct_lst(basefrom + tfv_file, tf_d_l, wpg_l, save_encoding, error_handling):
            print("TF Vector(s) File %s : Saved" % tfv_file)
            res = [ "TF Vector(s) File %s : Saved" % tfv_file ]
        else:
            print("TF Vector(s) File %s : Error Occurred" % tfv_file) 
            res = [ "TF Vector(s) File %s : Error Occurred" % tfv_file ]
        #del wpg_l
        #print("Building Dictionary")
        #tf_dict = self.gen_tfd_frmlist(tf_d_l)
        #print("Dictionary Ready")
        #if self.save_tf_dct(basefrom + tfd_file, tf_dict):
        #    print("TF-Dictionary File %s : Saved" % tfd_file)
        #    res.append( "TF-Dictionary File %s : Saved" % tfd_file )
        #else: 
        #    print("TF-Dictionary File %s : Saved" % tfd_file)
        #    res.append( "TF-Dictionary File %s : Error Occurred" % tfd_file )
        return res
    

class HTML2NF(VHTools, RegexHTML2NgFTools):
    
    def __init__(self, n=3):
        RegexHTML2NgFTools.__init__(self, n)
    
    def html2nfv_n_nfd(self, basefrom_l, pathsfrom, nfv_file, nfd_file, err_file, load_encoding='utf-8', save_encoding='utf-8', error_handling='strict', low_mem='True'):
        #Find witch base path is present
        if isinstance(basefrom_l, list):
            for basefrom in basefrom_l:
                filepath = basefrom + pathsfrom    
                flist = [files for path, dirs, files in os.walk(filepath)]
                if flist:
                    break
        else:
            basefrom = basefrom_l
        print filepath 
        wpg_l, nf_d_l = self.nf_d_frmpaths(basefrom, pathsfrom, load_encoding, error_handling, low_mem)
        rmv_wpgs_l = list()
        rmv_idxs_l = list()
        with codecs.open(basefrom + '/' + err_file, 'w', 'utf-8', 'strict') as  fenc:      
            for i in range(len(wpg_l)):
                if not nf_d_l[i]:
                    fenc.write( str(wpg_l[i]) + ' => ' + str(nf_d_l[i]) + '\n' )
                    rmv_wpgs_l.append( wpg_l[i] )
                    rmv_idxs_l.append( i )
        for i, rmv_i in enumerate(rmv_idxs_l):
            nf_d_l.pop( rmv_i - i )
        for wpg in rmv_wpgs_l:
            wpg_l.remove( wpg )
        if self.save_tf_dct_lst(basefrom + nfv_file, nf_d_l, wpg_l, save_encoding, error_handling):
            print("NgF Vector(s) File %s : Saved" % nfv_file)
            res = [ "NgF Vector(s) File %s : Saved" % nfv_file ]
        else:
            print("NgF Vector(s) File %s : Error Occurred" % nfv_file) 
            res = [ "NgF Vector(s) File %s : Error Occurred" % nfv_file ]
        return res

         
        