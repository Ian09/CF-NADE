'''
Created on Dec 2, 2015

@author: yin.zheng
'''

import os
from fuel.datasets import H5PYDataset
class MovieLens10M(H5PYDataset):
    def __init__(self, which_set, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        self.filename = 'movielens-10m.hdf5'
#         self.sources = ('input_ratings', 'output_ratings', 'input_masks', 'output_masks')
        super(MovieLens10M, self).__init__(self.data_path,
                                          which_set,
                                          **kwargs)
    @property
    def data_path(self):
        path = os.path.join(os.environ['MovieLens10M'],self.filename)
        return path
    
    
    
    