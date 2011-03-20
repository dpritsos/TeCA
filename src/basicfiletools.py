"""

"""

import codecs
import os
import sys
#from scgenrelerner_svmbased import *


class BaseFileTools(object):
    
    @staticmethod
    def file_list_frmpaths(basepath, filepath_l):
        if basepath is None:
            basepath = '' 
        if isinstance(filepath_l, str):
            flist = [ files_n_paths[2] for files_n_paths in os.walk( basepath + filepath_l ) ]
            flist = flist[0]
            fname_lst = [ basepath + filepath_l + fname for fname in flist ]
        elif isinstance(filepath_l, list):
            fname_lst = list()
            for filepath in filepath_l:
                flist = [ files_n_paths[2] for files_n_paths in os.walk( basepath + filepath ) ]
                flist = flist[0]
                fname_lst.extend( [ basepath + filepath + fname for fname in flist ] )
        else:
            raise Exception("A String or a list of Strings was Expected as input - Stings should be file-paths")
        return fname_lst
    
    @staticmethod
    def copyfile(source, dest):
        """ copyfile(): Copy a file from source to dest path. """
        source_f = open(source, 'rb')
        dest_f = open(dest, 'wb')
        dest_path = os.path.split(dest)
        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)
        while True:
            copy_buffer = source_f.read(1024*1024)
            if copy_buffer:
                dest_f.write(1024*1024)
            else:
                break
        source_f.close()
        dest_f.close()
        
    @staticmethod
    def movefile(source, dest):
        """ movefile(): A UNIX compatible function for moving file from Source path
            to Destination path. The Source path Hard Link is deleted  """
        os.link(source, dest)
        os.unlink(source)
        

class PathCleanUp(BaseFileTools):
    
    @staticmethod
    def path_lst_frmfile(path_lst_str):
        """
        """
        with codecs.open(path_lst_str, 'rb', 'utf-8', 'strict') as fenc: 
            filepath_lst = list()
            for line in fenc:
                line.rstrip()
                filepath = line.split(' => ')   
                filepath_lst.append( filepath[0] )
        return filepath_lst
    
    @staticmethod
    def move_flst(flist_filepath, dest, basepath=None):
        path_lst = PathCleanUp.path_lst_frmfile(flist_filepath)
        if basepath:
            if not os.path.isdir( os.path.join(basepath, dest) ):
                os.mkdir( os.path.join(basepath, dest) )
            for fpath in path_lst:
                fpath_spled = os.path.split(fpath)
                dest_file = os.path.join(basepath, dest, fpath_spled[1])
                PathCleanUp.movefile(fpath, dest_file)
        else:
            if not os.path.isdir( os.path.join(dest) ):
                os.mkdir(dest) 
            for fpath in path_lst:
                fpath_spled = os.path.split(fpath)
                dest_file = os.path.join(dest, fpath_spled[1])
                PathCleanUp.movefile(fpath, dest_file)
        
    @staticmethod
    def move_atleast(sourcepath, dest, famount, basepath=''):
        fpath_lst = PathCleanUp.file_list_frmpaths(None, sourcepath)
        if not os.path.isdir( os.path.join(basepath, dest) ):
            os.mkdir( os.path.join(basepath, dest) )
        fpath_lst.sort() #Maybe there is no need for this line of code
        last_fname = ''
        fappnd_cnt = 0
        for fpath in fpath_lst:
            fpath_spled = os.path.split( fpath )
            fpath_spled = fpath_spled[1]
            fname_sld = fpath_spled.split('.')
            fname = '.'.join(fname_sld[:-2])
            if fname == last_fname:
                PathCleanUp.movefile( fpath, os.path.join(dest, fpath_spled) )
                fappnd_cnt += 1
            else:
                if fappnd_cnt < famount:
                    last_fname = fname
                else:
                    break
    
    @staticmethod
    def move_frmto(sourcepath, dest, famount, basepath=''):
        fpath_lst = PathCleanUp.file_list_frmpaths(None, sourcepath)
        if not os.path.isdir( os.path.join(basepath, dest) ):
            os.mkdir( os.path.join(basepath, dest) )
        fpath_lst.sort() #Maybe there is no need for this line of code
        fappnd_cnt = 0
        for fpath in fpath_lst:
            fpath_spled = os.path.split( fpath )
            fpath_spled = fpath_spled[1]
            PathCleanUp.movefile( fpath, os.path.join(dest, fpath_spled) )
            fappnd_cnt += 1
            if fappnd_cnt >= famount:
                break
    
    
#Unit Test    
if __name__ == '__main__':
    
    base_filepath = "/home/dimitrios/Documents/Synergy-Crawler/KI-04"
    #base_filepath = "/home/dimitrios/Documents/Synergy-Crawler/Santini_corpus"
    #base_filepath = "/home/dimitrios/Documents/Synergy-Crawler/saved_pages"
    genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop" ] 
    #genres = [ "blogs", "news" , "product_companies", "forum" ] #"wiki_pages",
    #genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]      
    
    #for g in genres:
    #    flist_filepath = os.path.join(base_filepath, g, g + ".train.lst.err" )
    #    dest = os.path.join(base_filepath, g, "non_proper_html_pages")
    #    PathCleanUp.move_flst(flist_filepath, dest)
    
    
    #for g in genres: 
    #    sourcepath = os.path.join(base_filepath, g, "html_pages/" )
    #    dest = os.path.join(base_filepath, g, "test_only_html_pages/" )
    #    print dest
    #    PathCleanUp.move_atleast(sourcepath, dest, 1000, basepath='')
        
    for g in genres: 
        sourcepath = os.path.join(base_filepath, g, "html_pages/" )
        dest = os.path.join(base_filepath, g, "test_only_html_pages/" )
        PathCleanUp.move_frmto(sourcepath, dest, 50, basepath='')
        
    print("Thank you and Goodbye!!!")
                
            
        
            
            
    
    
    
    
        
    
