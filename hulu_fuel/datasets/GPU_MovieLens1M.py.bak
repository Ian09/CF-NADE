'''
Created on Dec 2, 2015

@author: yin.zheng
'''

import os
# from fuel.datasets import H5PYDataset
import numpy as np

def load(seed = 1234):
    
    np.random.seed(seed)
    n_users = 6040
    n_movies = 3952
    shuffle_order = np.random.permutation(n_users)
    
    train_ratings = np.load(os.path.join(os.environ['MovieLens1M'], 'train.npy'))[shuffle_order]
    valid_ratings = np.load(os.path.join(os.environ['MovieLens1M'], 'valid.npy'))[shuffle_order]
    test_ratings = np.load(os.path.join(os.environ['MovieLens1M'], 'test.npy'))[shuffle_order]
    
    def preprocess_data(ratings_2d):
        
        input_ratings =  ratings_2d
        input_shape = input_ratings.shape
        K = 5
        input_ratings_3d = np.zeros((input_shape[0], input_shape[1], K), 'int8')
        input_ratings_nonzero = input_ratings.nonzero()
        input_ratings_3d[input_ratings_nonzero[0],
                         input_ratings_nonzero[1],
                         input_ratings[input_ratings_nonzero[0],
                                       input_ratings_nonzero[1]
                                       ] - 1] = 1
                                       
        
        
        return input_ratings_3d
    
    train_ratings_3d = preprocess_data(train_ratings)
    valid_ratings_3d = preprocess_data(valid_ratings)
    test_ratings_3d = preprocess_data(test_ratings)
    
    return train_ratings_3d, valid_ratings_3d, test_ratings_3d
    
    
    
    
    
if __name__ == '__main__':
    
    print 123
    
    