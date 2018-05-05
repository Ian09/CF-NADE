'''
Created on Dec 2, 2015

@author: yin.zheng
'''

import os
from fuel.datasets import H5PYDataset
import numpy as np
class MovieLens1M(H5PYDataset):
    def __init__(self, which_set, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        self.filename = 'movielens-1m.hdf5'
#         self.sources = ('input_ratings', 'output_ratings', 'input_masks', 'output_masks')
        super(MovieLens1M, self).__init__(self.data_path,
                                          which_set,
                                          **kwargs)
    @property
    def data_path(self):
        path = os.path.join(os.environ['MovieLens1M'],self.filename)
        return path
    
    
class foo(object):
    
    def __init__(self):
        self.a = 1000
        
    @property
    def voltage(self):
        return 4
    
if __name__ == '__main__':
    
    f = foo()
    print f.voltage
    print 123
    
    