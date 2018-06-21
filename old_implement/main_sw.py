import tensorflow as tf
import numpy as np

from Data_timestamp import Data_user
from CF_NADE_Time_Stamp import CF_NADE


flags = tf.app.flags

"""
define the parameters
"""
flags.DEFINE_float('time_function_lambda',0.2,'time function parameter');
flags.DEFINE_float('time_transform_parameter', 0, 'for the weights decay through time')
flags.DEFINE_integer('batch_size', 512, 'batch_size for the users')
flags.DEFINE_integer('movie_dim', 3706, 'how many movies in the dataset')
flags.DEFINE_integer('num_classes', 5, 'score range')
flags.DEFINE_float('learning_rate', 0.001, 'learning_rate for Adam')
flags.DEFINE_integer('hidden_dim', 500, 'dimenstion of hidden states')
flags.DEFINE_integer('epochs', 500, 'epochs to train')
flags.DEFINE_integer('test_avg_num', 1, 'how many test acc should we get to averge final test acc')
flags.DEFINE_float('weight_decay', 1, 'weight decay for regularization')
flags.DEFINE_float('weight_W', 1, 'weight W')
flags.DEFINE_float('weight_OUT_W', 0, 'weight OUT_W')
flags.DEFINE_integer('embedding_dim', 100, '')
flags.DEFINE_integer('seq_len', 20, '')
FLAGS = flags.FLAGS


if __name__ == '__main__':
	log = open('test_eval.dat', 'a+')
	movie_dim = 3706   
	num_classes = 5
	hidden_dim = 500
	W_array = np.zeros((3706, 5, 500))
	Output_W_array = np.zeros((500, 3706, 5))
	b_array = np.zeros((500))
	Out_b_array = np.zeros((3706, 5))
	W_array = np.load("save_value/W.data.npy")
	Out_W_array = np.load("save_value/OUTW.data.npy")
	b_array = np.load("save_value/B.data.npy")
	Out_b_array = np.load("save_value/OUTB.data.npy")
	myData = Data_user('../ml-1m/ratings.dat')
	myData.split_set(0.9, 0)
	myData.prepare_data()

	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

    
	with tf.Session(config=run_config) as sess:
		cf_nade = CF_NADE(sess, FLAGS)
		cf_nade.time_stamp_test(myData,FLAGS,log,W_array,Output_W_array,b_array,Out_b_array)       
    
	print("Yes!")
	tf.app.run()