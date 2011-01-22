
import os
from vectorhandlingtools import VHTools 
from html2vector import RegexHTML2NgFTools, RegexHTML2TFTools

class HTML2TF(VHTools, RegexHTML2TFTools):
    
    def __init__(self):
        RegexHTML2TFTools.__init__(self)
    
    def html2tfv_n_tfd(self, basefrom_l, pathsfrom, tfv_file, tfd_file, encoding='utf8', error_handling='strict', low_mem='True'):
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
        wpg_l, tf_d_l = self.tf_d_frmpaths(basefrom, pathsfrom, encoding, error_handling, low_mem)
        rmv_wpgs_l = list()
        for i in range(len(wpg_l)):
            if tf_d_l[i] is None:
                print wpg_l[i], tf_d_l[i]
                #TO BE SAVED in a file
                rmv_wpgs_l.append( wpg_l[i] )
        for i in range(len(rmv_wpgs_l)):
            tf_d_l.remove( None )
        for wpg in rmv_wpgs_l:
            wpg_l.remove( wpg )
        if self.save_tf_dct_lst(basefrom + tfv_file, tf_d_l, wpg_l):
            res = [ "TF Vector(s) File %s : Saved" % tfv_file ]
        else: 
            res = [ "TF Vector(s) File %s : Error Occurred" % tfv_file ]
        del wpg_l
        tf_dict = self.gen_tfd_frmlist(tf_d_l)
        if self.save_tf_dct(basefrom + tfd_file, tf_dict):
            res.append( "TF-Dictionary File %s : Saved" % tfd_file )
        else: 
            res.append( "TF-Dictionary File %s : Error Occurred" % tfd_file )
        return res
    

class HTML2NgF(VHTools, RegexHTML2NgFTools):
    
    def __init__(self, n=3):
        RegexHTML2NgFTools.__init__(self, n)
    
    def htmlngfv_n_ngfd(self, basefrom_l, pathsfrom, ngfv_file, ngfd_file, encoding='utf8', error_handling='strict', low_mem='True'):
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
        wpg_l, ngf_d_l = self.tf_d_frmpaths(basefrom, pathsfrom, encoding, error_handling, low_mem)
        rmv_wpgs_l = list()
        for i in range(len(wpg_l)):
            if ngf_d_l[i] is None:
                print wpg_l[i], ngf_d_l[i]
                #TO BE SAVED in a file
                rmv_wpgs_l.append( wpg_l[i] )
        for i in range(len(rmv_wpgs_l)):
            ngf_d_l.remove( None )
        for wpg in rmv_wpgs_l:
            wpg_l.remove( wpg )
        if self.save_tf_dct_lst(basefrom + ngfv_file, ngf_d_l, wpg_l):
            res = [ "NF Vector(s) File %s : Saved" % ngfv_file ]
        else: 
            res = [ "NF Vector(s) File %s : Error Occurred" % ngfv_file ]
        del wpg_l
        tf_dict = self.gen_tfd_frmlist(ngf_d_l)
        if self.save_tf_dct(basefrom + ngfd_file, tf_dict):
            res.append( "NF-Dictionary File %s : Saved" % ngfd_file )
        else: 
            res.append( "NF-Dictionary File %s : Error Occurred" % ngfd_file )
        return res    


         
        