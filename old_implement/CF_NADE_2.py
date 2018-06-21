import tensorflow as tf
import numpy as np
import os

class CF_NADE():
	def __init__(self, sess, flags):
		self.sess = sess
		self.X = tf.placeholder("float32",[None, flags.movie_dim]) # batch * movie_dim 
		self.input_mask = tf.placeholder('float32', [None, flags.movie_dim])
		self.output_mask = tf.placeholder('float32', [None, flags.movie_dim])

		self.weights = {
		'W': tf.Variable(tf.random_normal([flags.movie_dim, flags.num_classes, flags.hidden_dim])),
		'Output_W' : tf.Variable(tf.random_normal([flags.hidden_dim, flags.movie_dim, flags.num_classes]))
		}

		self.bias = {
	    'b_hidden':tf.Variable(tf.random_normal([flags.hidden_dim])),
	    'b_output':tf.Variable(tf.random_normal([flags.movie_dim, flags.num_classes]))
		}

	def buildGraph(self, flags):
		#transform input_X to batch*movie_dim*num_classes

		self.int_X = tf.one_hot(tf.cast(self.X - 1, dtype=tf.int32), axis=-1, depth=flags.num_classes)
		#self.X_cum = tf.cumsum(self.int_X, axis=-1, reverse=True)

		self.input_mask_1 = tf.expand_dims(self.input_mask, 2)
		self.input_mask_2 = tf.tile(self.input_mask_1, [1,1,flags.num_classes])
		self.output_mask_1 = tf.expand_dims(self.output_mask, 2)
		self.output_mask_2 = tf.tile(self.output_mask_1, [1,1,flags.num_classes])
		#dim(self.H) = batch_size * hidden_dim
		self.H = tf.tanh(tf.add(self.bias['b_hidden'],tf.tensordot(tf.multiply(self.int_X, self.input_mask_2), self.weights['W'], axes=[[1,2],[0,1]])))
		#dim(self.output_layer) = batch_size * movie_dim * num_classes
		self.output_layer = tf.multiply(tf.add(self.bias['b_output'], tf.tensordot(self.H, self.weights['Output_W'], axes=[[1],[0]])), self.output_mask_2)

		#self.output_layer = tf.cumsum(self.output_layer,axis=2)   #Cumsum for output_layer
		self.scores_prob = tf.nn.softmax(self.output_layer, axis=2)

		#dim(self.pred_scores) = batch * movie
		self.pred_scores = tf.tensordot(self.scores_prob, [1.0,2.0,3.0,4.0,5.0], axes=[[2],[0]])

		self.true_scores = tf.tensordot(tf.multiply(self.int_X,self.output_mask_2), [1.0,2.0,3.0,4.0,5.0], axes=[[2],[0]])
        
		#dim(self.true_scores_onehot) = batch * movie * num_classes
		self.true_scores_onehot = tf.one_hot(tf.cast(self.true_scores - 1, tf.int32), axis=-1, depth=flags.num_classes)

		self.loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.true_scores_onehot, logits=self.output_layer))
		#regularization
		self.loss_op += flags.weight_decay * (flags.weight_W * tf.reduce_sum(self.weights['W'] * self.weights['W']) + flags.weight_OUT_W * tf.reduce_sum(self.weights['Output_W'] * self.weights['Output_W']))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=flags.learning_rate, beta1=0.1, beta2=0.001)
		#self.optimizer = tf.train.AdamOptimizer(learning_rate=flags.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss_op)

		self.eval = tf.sqrt(tf.losses.mean_squared_error(labels=self.true_scores , predictions=self.pred_scores, weights = self.output_mask))
		self.init = tf.global_variables_initializer()


	def train(self, myData, flags):
		self.buildGraph(flags)

		self.sess.run(self.init)	

		for epoch in range(flags.epochs):
			print ('#####################Epoch:', epoch,'########################\n')
			X, output_mask, input_mask, flag = myData.get_batch_train(512)
			while(flag):
				self.sess.run([self.train_op], feed_dict={self.X:X, self.input_mask:input_mask, self.output_mask:output_mask})                
				X, input_mask, output_mask, flag = myData.get_batch_train(512)
			myData.renew_train()
			myData.shuffle_data()
			# if not os.path.exists('./checkpoints'):
			# 	os.makedirs('./checkpoints')
			# saver.save(self.sess, './checkpoints/CF_NADE.model', global_step=batch)

	def test(self, myData, flags, log):
		X, output_mask, input_mask, flag = myData.get_batch_test(512)
		acc = []
		while(flag):
			myEval = self.sess.run(self.eval, feed_dict={self.X:X, self.input_mask:input_mask, self.output_mask:output_mask})
			print ('test_acc:', myEval)
			acc.append(myEval)
			X, output_mask, input_mask, flag = myData.get_batch_test(512)
		print ('final test_acc:', np.mean(acc))
		myData.renew_test()
		log.write(str(np.mean(acc)) + '\n')