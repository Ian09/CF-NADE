import tensorflow as tf
import numpy as np
from CF_NADE_2 import CF_NADE
from Data_user_per_sample import Data_user
# coding=utf-8
flags = tf.app.flags

def main(_):
	tf.reset_default_graph()
	W = tf.get_variable('W',[3706, 5, 500])
	saver = tf.train.Saver()
	with tf.Session() as sess:
		save_path = "save_value/variable.ckpt"
		saver.restore(sess, save_path) 
		print("v1 : $s" % W.eval());
        
if __name__ == '__main__':
	tf.app.run()
