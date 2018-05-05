import tensorflow as tf
import numpy as np
from CF_NADE import CF_NADE
from Data_user_per_sample import Data_user

flags = tf.app.flags

"""
define the parameters
"""
flags.DEFINE_float('time_transform_parameter', 0, 'for the weights decay through time')
flags.DEFINE_integer('batch_size', 512, 'batch_size for the users')
flags.DEFINE_integer('movie_dim', 3706, 'how many movies in the dataset')
flags.DEFINE_integer('num_classes', 5, 'score range')
flags.DEFINE_float('learning_rate', 0.001, 'learning_rate for Adam')
flags.DEFINE_integer('hidden_dim', 500, 'dimenstion of hidden states')
flags.DEFINE_boolean('train', False, 'whether to train model')
flags.DEFINE_integer('epochs', 10, 'epochs to train')
flags.DEFINE_float('weight_decay', 1, 'weight decay for regularization')
FLAGS = flags.FLAGS

def main(_):
	myData = Data_user('../ml-1m/ratings.dat')
	myData.split_set(0.9)

	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		cf_nade = CF_NADE(sess, FLAGS)

		if (FLAGS.train == True):
			cf_nade.train(myData, FLAGS)



if __name__ == '__main__':
	tf.app.run()