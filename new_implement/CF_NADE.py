import tensorflow as tf
import numpy as np
import os

class CF_NADE():
	def __init__(self, sess, flags):
		self.sess = sess
		self.X = tf.placeholder("float32",[None, flags.movie_dim, flags.num_classes]) # batch * movie_dim * num_classes
		self.Y = tf.placeholder("int32",[None, 2]) # batch * (ratings, idex)

		self.weights = {
		'W': tf.Variable(tf.random_normal([flags.movie_dim, flags.num_classes, flags.hidden_dim])),
		'Output_W' : tf.Variable(tf.random_normal([flags.hidden_dim, flags.movie_dim, flags.num_classes]))
		}

		self.bias = {
	    'b_hidden':tf.Variable(tf.random_normal([flags.hidden_dim])),
	    'b_output':tf.Variable(tf.random_normal([flags.movie_dim, flags.num_classes]))
		}

	def buildGraph(self, flags):
		#dim(self.H) = batch_size * hidden_dim
		self.H = tf.tanh(tf.add(self.bias['b_hidden'],tf.tensordot(self.X, self.weights['W'], axes=[[1,2],[0,1]])))
		#dim(self.output_layer) = batch_size * movie_dim * num_classes
		self.output_layer = tf.add(self.bias['b_output'], tf.tensordot(self.H, self.weights['Output_W'], axes=[[1],[0]]))
		#transform ouput_layer to dim=batch_size*num_classes
		self.one_hot = tf.one_hot(self.Y[:,1], depth=flags.movie_dim)
		self.one_hot = tf.expand_dims(self.one_hot, 2)
		self.one_hot = tf.tile(self.one_hot, [1,1,flags.num_classes])
		#dim(self.output_layer) = batch_size * num_classes
		self.output_layer = tf.reduce_sum(self.output_layer * self.one_hot, axis=1)

		#calculate pred_scores
		self.scores_prob = tf.nn.softmax(self.output_layer, axis=1)
		scores_matrix = np.repeat([[1,2,3,4,5]], flags.batch_size, axis=0)
		scores_matrix = tf.convert_to_tensor(scores_matrix, tf.float32)
		self.pred_scores = tf.reduce_sum(scores_matrix * self.scores_prob, axis=1)

		self.true_scores = self.Y[:, 0]
		self.true_scores_onehot = tf.one_hot(self.true_scores - 1, depth=flags.num_classes)

		self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.true_scores_onehot, logits=self.output_layer))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=flags.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss_op)

		self.eval = tf.losses.mean_squared_error(labels=self.true_scores, predictions=self.pred_scores)
		self.init = tf.global_variables_initializer()

	def data_transform(self, X, flags):
		#dim(X) = batch_size * num_classes * (ratings, time_diff)
		get_ratings = X[:,:,0]
		ratings_X = np.zeros((flags.batch_size ,flags.movie_dim, flags.num_classes)) #num_classes:0,1,2,3,4 -> ratings:1,2,3,4,5
		for iter_num_batch in range(flags.batch_size):
			for iter_movie in range(flags.movie_dim):
				#if get_ratings[iter_num_batch,iter_movie] > 0:
					#for fill_in_iter in range(int(get_ratings[iter_num_batch,iter_movie])):
					#	ratings_X[iter_num_batch,iter_movie,fill_in_iter]=self.time_stamp_function_exp(X[iter_num_batch,iter_movie,1],flags.time_transform_parameter)
                        
				if get_ratings[iter_num_batch,iter_movie] > 0:
					ratings_X[iter_num_batch,iter_movie,int(get_ratings[iter_num_batch,iter_movie] - 1)] = self.time_stamp_function_exp(X[iter_num_batch,iter_movie,1],flags.time_transform_parameter)
		
		return ratings_X

	def time_stamp_function_exp(self, dif_time_stamp,time_transform_parameter):
	    return np.exp(-abs(dif_time_stamp * time_transform_parameter))

	def train(self, myData, flags):
		self.buildGraph(flags)

		self.sess.run(self.init)	

		saver = tf.train.Saver()
		print ('reading checkpoints....')
		ckpt = tf.train.get_checkpoint_state('./checkpoints/')
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			saver.restore(sess, os.path.join('./checkpoints', ckpt_name))
			print("[*] Success to read {}".format(ckpt_name))
		else:
			print('fail to read checkpoints, begin to train....')

		batch = 1
		for epoch in range(5):
			print ('Epoch:', epoch)
			while (True):
				try:
					X, Y = myData.get_batch_new_2(512, 'train')
				except:
					break
				X = self.data_transform(X, flags)
				pred_scores, _ = self.sess.run([self.pred_scores, self.train_op], feed_dict={self.X:X, self.Y:Y})
				print (batch)
				batch += 1
				if (batch % 5 == 1):
					myEval = self.sess.run(self.eval, feed_dict={self.X:X, self.Y:Y})
					print ('eval:', myEval)
				if (batch % 10 == 1):
					myLoss = self.sess.run(self.loss_op, feed_dict={self.X:X, self.Y:Y})
					print ('loss:', myLoss)
			myData.renew()
			if not os.path.exsits('./checkpoints'):
				os.makedirs('./checkpoints')
			saver.save(sess, './checkpoints/CF_NADE.model', global_step=count)

    