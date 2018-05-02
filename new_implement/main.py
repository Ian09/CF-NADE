import tensorflow as tf
import numpy as np
from CF_NADE import CF_NADE

flags = tf.app.flags

"""
define the parameters
"""
flags.DEFINE_float('time_transform_parameter', 0.1, 'for the weights decay through time')
flags.DEFINE_integer('batch_size', 512, 'batch_size for the users')
flags.DEFINE_integer('movie_dim', None, 'how many movies in the dataset')
flags.DEFINE_integer('num_classes', 5, 'score range')
flags.DEFINE_float('learning_rate', 0.1, 'learning_rate for Adam')
flags.DEFINE_integer('hidden_dim', 500, 'dimenstion of hidden states')
flags.DEFINE_boolean('train', False, 'whether to train model')
flags.DEFINE_integer('epochs', 10, 'epochs to train')
FLAGS = flags.FLAGS

def data_transform(X, flags):
	#dim(X) = batch_size * num_classes * (ratings, time_diff)
    get_ratings = X[:,:,0]
    ratings_X = np.zeros((flags.batch_size ,movie_dim, num_classes)) #num_classes:0,1,2,3,4 -> ratings:1,2,3,4,5
    for iter_num_batch in range(flags.batch_size):
        for iter_movie in range(flags.movie_dim):
            if get_ratings[iter_num_batch,iter_movie] > 0:
                ratings_X[iter_num_batch,iter_movie,int(get_ratings[iter_num_batch,iter_movie] - 1)] = time_stamp_function_exp(X[iter_num_batch,iter_movie,1],time_transform_parameter)
    return ratings_X

def time_stamp_function_exp(dif_time_stamp,time_transform_parameter):
    return np.exp(-abs(dif_time_stamp * time_transform_parameter))

def main():
	#get data X, Y first

	X = data_transform(X, FLAGS)

	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		cf_nade = CF_NADE(sess, FLAGS)

		if (FLAGS.train == True):
			cf_nade.train(X, Y, FLAGS)



if __name__ == '__main__':
	tf.app.run()