'''
Created on Dec 29, 2015

@author: yin.zheng
'''
from scipy.sparse import csr_matrix
import numpy as np
import os

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data , indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])
    
    
        
def load(which_set):
    
    data_path = os.path.join(os.environ['Netflix'])
    input_ratings = load_sparse_csr(os.path.join(data_path, which_set + '_input_ratings.npz'))
    input_masks = load_sparse_csr(os.path.join(data_path, which_set + '_input_masks.npz'))
    output_ratings = load_sparse_csr(os.path.join(data_path, which_set + '_output_ratings.npz'))
    output_masks = load_sparse_csr(os.path.join(data_path, which_set + '_output_masks.npz'))
    
    return input_ratings, output_ratings, input_masks, output_masks
        
