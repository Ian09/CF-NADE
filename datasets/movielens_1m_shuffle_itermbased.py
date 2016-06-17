'''
Created on Dec 1, 2015

@author: yin.zheng
'''

import numpy as np
import scipy.io as sio
import os
import h5py
import cv2
from scipy.sparse import csr_matrix, lil_matrix
import random

def read_ratings(filename):
    ratings = []
    with open(filename) as fp:
        for line in fp:
            user_id, mov_id, rating, time = line.split('::')
            ratings.append([int(user_id), int(mov_id), int(rating), int(time)])
    return ratings

def read_users(filename):
    users = []
    with open(filename) as fp:
        for line in fp:
            UserID, Gender, Age, Occupation, Zip_code = line.split('::')
            users.append([int(UserID), Gender, int(Age), int(Occupation), Zip_code])
    return users

def read_movies(filename):
    movies = []
    with open(filename) as fp:
        for line in fp:
            MovieID, Title, Genres = line.split('::')
            movies.append([int(MovieID), Title, Genres])
    return movies

def write_movie_data(ratings, data_path, output, seed):
    
    users = {}
    movs = {}
    cnt_u = 0
    cnt_i = 0
    for user_id, mov_id, rating, _ in ratings:
        if user_id not in users.keys():
            users[user_id] = cnt_u
            cnt_u += 1
        if mov_id not in movs.keys():
            movs[mov_id] = cnt_i
            cnt_i += 1
    n_users = len(users)
    n_movies = len(movs)
    train_ratio = 0.9*0.995
    valid_ratio = 0.9*0.005
    test_ratio = 0.1
    n_ratings = len(ratings)
    n_test = np.ceil(n_ratings*test_ratio)
    n_valid = np.ceil(n_ratings*valid_ratio)
    n_train = n_ratings - n_test - n_valid
    
    train_input_ratings = np.zeros((n_movies, n_users), dtype='int8')
    train_output_ratings = np.zeros((n_movies, n_users), dtype='int8')
    train_input_masks = np.zeros((n_movies, n_users), dtype='int8')
    train_output_masks = np.zeros((n_movies, n_users), dtype='int8')
    
    valid_input_ratings = np.zeros((n_movies, n_users), dtype='int8')
    valid_output_ratings = np.zeros((n_movies, n_users), dtype='int8')
    valid_input_masks = np.zeros((n_movies, n_users), dtype='int8')
    valid_output_masks = np.zeros((n_movies, n_users), dtype='int8')
    
    test_input_ratings = np.zeros((n_movies, n_users), dtype='int8')
    test_output_ratings = np.zeros((n_movies, n_users), dtype='int8')
    test_input_masks = np.zeros((n_movies, n_users), dtype='int8')
    test_output_masks = np.zeros((n_movies, n_users), dtype='int8')
    
    
    random.seed(seed)
    random.shuffle(ratings)
    total_n_train = 0
    total_n_valid = 0
    total_n_test = 0
    cnt = 0
    for user_id, mov_id, rating, _ in ratings:
        if cnt < n_train:
            train_input_ratings[movs[mov_id], users[user_id]] = rating
            train_input_masks[movs[mov_id], users[user_id]] = 1
            valid_input_ratings[movs[mov_id], users[user_id]] = rating
            valid_input_masks[movs[mov_id], users[user_id]] = 1
            total_n_train += 1
        elif cnt < n_train+n_valid:
            valid_output_ratings[movs[mov_id], users[user_id]] = rating
            valid_output_masks[movs[mov_id], users[user_id]] = 1
            total_n_valid += 1
        else:
            test_output_ratings[movs[mov_id], users[user_id]] = rating
            test_output_masks[movs[mov_id], users[user_id]] = 1
            total_n_test += 1
        cnt += 1
    test_input_ratings = train_input_ratings + valid_output_ratings
    test_input_masks = train_input_masks + valid_output_masks        
    
#     rating_mat = csr_matrix(rating_mat)
    
    input_r = np.vstack((train_input_ratings, valid_input_ratings, test_input_ratings))
    input_m = np.vstack((train_input_masks, valid_input_masks, test_input_masks))
    output_r = np.vstack((train_output_ratings, valid_output_ratings, test_output_ratings))
    output_m = np.vstack((train_output_masks, valid_output_masks, test_output_masks))
    
    
    f = h5py.File(os.path.join(output, 'movielens-1m.hdf5'), 'w')
    input_ratings = f.create_dataset('input_ratings', shape=(n_movies*3, n_users), dtype='int8', data=input_r)
    input_ratings.dims[0].label = 'batch'
    input_ratings.dims[1].label = 'movies'
    input_masks = f.create_dataset('input_masks', shape=(n_movies*3, n_users), dtype='int8', data=input_m)
    input_masks.dims[0].label = 'batch'
    input_masks.dims[1].label = 'movies'
    output_ratings = f.create_dataset('output_ratings', shape=(n_movies*3, n_users), dtype='int8', data=output_r)
    output_ratings.dims[0].label = 'batch'
    output_ratings.dims[1].label = 'movies'
    output_masks = f.create_dataset('output_masks', shape=(n_movies*3, n_users), dtype='int8', data=output_m)
    output_masks.dims[0].label = 'batch'
    output_masks.dims[1].label = 'movies'
    
    split_array = np.empty(
                           12,
                           dtype=([
                                   ('split', 'a', 5),
                                   ('source', 'a', 14),
                                   ('start', np.int64, 1),
                                   ('stop', np.int64, 1),
                                   ('indices', h5py.special_dtype(ref=h5py.Reference)),
                                   ('available', np.bool, 1),
                                   ('comment', 'a', 1)
                                   ]
                                  )
                           )
    split_array[0:4]['split'] = 'train'.encode('utf8')
    split_array[4:8]['split'] = 'valid'.encode('utf8')
    split_array[8:12]['split'] = 'test'.encode('utf8')
    split_array[0:12:4]['source'] = 'input_ratings'.encode('utf8')
    split_array[1:12:4]['source'] = 'input_masks'.encode('utf8')
    split_array[2:12:4]['source'] = 'output_ratings'.encode('utf8')
    split_array[3:12:4]['source'] = 'output_masks'.encode('utf8')
    split_array[0:4]['start'] = 0
    split_array[0:4]['stop'] = n_movies
    split_array[4:8]['start'] = n_movies
    split_array[4:8]['stop'] = n_movies*2
    split_array[8:12]['start'] = n_movies*2
    split_array[8:12]['stop'] = n_movies*3
    split_array[:]['indices'] = h5py.Reference()
    split_array[:]['available'] = True
    split_array[:]['comment'] = '.'.encode('utf8')
    f.attrs['split'] = split_array
    f.flush()
    f.close()
    
    f = open(os.path.join(output, 'metadata'), 'w')
    line = 'n_users:%d\n'%n_users
    f.write(line)
    line = 'n_movies:%d'%n_movies
    f.write(line)
    f.close()
    
    f = open(os.path.join(output, 'user_dict'), 'wb')
    import cPickle
    cPickle.dump(users, f)
    f.close()
    
    f = open(os.path.join(output, 'movie_dict'), 'wb')
    cPickle.dump(movs, f)
    f.close()
    
    


def main(data_path, output, seed):
    
    ratings = read_ratings(os.path.join(data_path, 'ratings.dat'))
#     movies = read_movies(os.path.join(data_path, 'movies.dat'))
#     users = read_users(os.path.join(data_path, 'users.dat'))
    
    write_movie_data(ratings, data_path, output, seed)

if __name__ == "__main__":
#     main("/Users/yin.zheng/Downloads/ml-1m",
#          "/Users/yin.zheng/ml_datasets/MovieLens1M-shuffle-itembased-0",
#          1234)
    print '1'
    main("/Users/yin.zheng/Downloads/ml-1m",
         "/Users/yin.zheng/ml_datasets/MovieLens1M-shuffle-itembased-1",
         2341)
    print '2'
    main("/Users/yin.zheng/Downloads/ml-1m",
         "/Users/yin.zheng/ml_datasets/MovieLens1M-shuffle-itembased-2",
         3412)
    print '3'
    main("/Users/yin.zheng/Downloads/ml-1m",
         "/Users/yin.zheng/ml_datasets/MovieLens1M-shuffle-itembased-3",
         4123)
    print '4'
    main("/Users/yin.zheng/Downloads/ml-1m",
         "/Users/yin.zheng/ml_datasets/MovieLens1M-shuffle-itembased-4",
         1324)
#     from fuel.datasets import H5PYDataset
#     
#     trainset = H5PYDataset(os.path.join('/Users/yin.zheng/ml_datasets/MovieLens1M-shuffle-itembased', 'movielens-1m.hdf5'),
#                            which_sets = ('train',),
#                            load_in_memory=True,
#                            sources=('input_ratings', 'output_ratings', 'input_masks', 'output_masks')
#                            )
#     print trainset.num_examples
#     from fuel.schemes import (SequentialScheme, ShuffledScheme,SequentialExampleScheme,ShuffledExampleScheme)
#     state = trainset.open()
#     scheme = ShuffledScheme(examples=trainset.num_examples, batch_size=3)
#     from fuel.streams import DataStream
#     data_stream = DataStream(dataset=trainset, iteration_scheme=scheme)
#     for data in data_stream.get_epoch_iterator():
#         print data[0].shape
    
    
    
