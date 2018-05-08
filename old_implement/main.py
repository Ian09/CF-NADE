import tensorflow as tf
import numpy as np
from CF_NADE_2 import CF_NADE
from Data_user_per_sample import Data_user
# coding=utf-8
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
flags.DEFINE_integer('epochs', 500, 'epochs to train')
flags.DEFINE_integer('test_avg_num', 1, 'how many test acc should we get to averge final test acc')
flags.DEFINE_float('weight_decay', 1, 'weight decay for regularization')
flags.DEFINE_float('weight_W', 1, 'weight W')
flags.DEFINE_float('weight_OUT_W', 0, 'weight OUT_W')
flags.DEFINE_integer('embedding_dim', 100, '')
flags.DEFINE_integer('seq_len', 20, '')
FLAGS = flags.FLAGS

def main(_):
	log = open('test_eval.dat', 'a+')
	log.write('weight_decay: %f, learning_rate: %f, weight_W: %f, weight_OUT_W: %f\n' % 
		(FLAGS.weight_decay, FLAGS.learning_rate, FLAGS.weight_W, FLAGS.weight_OUT_W))
	for i in range(FLAGS.test_avg_num):
		myData = Data_user('../ml-1m/ratings.dat')
		myData.split_set(0.9, 0)
		myData.prepare_data()

		run_config = tf.ConfigProto()
		run_config.gpu_options.allow_growth=True
        
		save_path = "save_value/"
		with tf.Session(config=run_config) as sess:
           
			cf_nade = CF_NADE(sess, FLAGS)

			cf_nade.train(myData, FLAGS)
			W_array = sess.run(cf_nade.weights['W']) 
			Out_W_array = sess.run(cf_nade.weights['Output_W'])
			b_array = sess.run(cf_nade.bias['b_hidden']) 
			Out_b_array = sess.run(cf_nade.bias['b_output']) 
			np.save("save_value/W.data",W_array)
			np.save("save_value/OUTW.data",Out_W_array)
			np.save("save_value/B.data",b_array)
			np.save("save_value/OUTB.data",Out_b_array)
			cf_nade.test(myData, FLAGS, log)

	log.close()

if __name__ == '__main__':
	tf.app.run()