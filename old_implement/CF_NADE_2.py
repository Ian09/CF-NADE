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
		self.loss_op += flags.weight_decay * (tf.reduce_sum(self.weights['W'] * self.weights['W']) + tf.reduce_sum(self.weights['Output_W'] * self.weights['Output_W']))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=flags.learning_rate, beta1=0.1, beta2=0.001)
		#self.optimizer = tf.train.AdamOptimizer(learning_rate=flags.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss_op)

		self.eval = tf.sqrt(tf.losses.mean_squared_error(labels=self.true_scores , predictions=self.pred_scores, weights = self.output_mask))
		self.init = tf.global_variables_initializer()


	def train(self, myData, flags):
		self.buildGraph(flags)

		self.sess.run(self.init)	

		# saver = tf.train.Saver()
		# print ('reading checkpoints....')
		# ckpt = tf.train.get_checkpoint_state('./checkpoints/')
		# if ckpt and ckpt.model_checkpoint_path:
		# 	ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		# 	saver.restore(self.sess, os.path.join('./checkpoints', ckpt_name))
		# 	print("[*] Success to read {}".format(ckpt_name))
		# else:
		# 	print('fail to read checkpoints, begin to train....')

		batch = 1
		log = open('result.dat', 'w')
		for epoch in range(flags.epochs):
			print ('#####################Epoch:', epoch,'########################\n')
			X, input_mask, output_mask, flag = myData.get_batch_train(512)
			while(flag):
				self.sess.run([self.train_op], feed_dict={self.X:X, self.input_mask:input_mask, self.output_mask:output_mask})
				print (batch)
				batch += 1
				if (batch % 5 == 1):
					myEval = self.sess.run(self.eval, feed_dict={self.X:X, self.input_mask:input_mask, self.output_mask:output_mask})
					print ('train_eval:', myEval)
					myLoss = self.sess.run(self.loss_op,feed_dict = {self.X:X,self.input_mask:input_mask,self.output_mask:output_mask})  
					print ('train_loss:',myLoss)
					#temp_pred_scores = self.sess.run(self.pred_scores,feed_dict = {self.X:X,self.input_mask:input_mask,self.output_mask:output_mask})      
					#print(temp_pred_scores)                    
				X, input_mask, output_mask, flag = myData.get_batch_train(512)

			loss = []
			count = 1
			test_X, input_mask, output_mask, flag = myData.get_batch_test(512)
			while(flag):
				if (count % 10) == 1:
					print (count)
				loss.append(self.sess.run([self.eval], feed_dict={self.X:test_X, self.input_mask:input_mask, self.output_mask:output_mask}))
				count += 1
				test_X, input_mask, output_mask, flag = myData.get_batch_test(512)
			print ('test eval:', np.mean(loss))
			log.write('test eval: %f' % np.mean(loss))
			myData.renew_test()
			myData.renew_train()
			# if not os.path.exists('./checkpoints'):
			# 	os.makedirs('./checkpoints')
			# saver.save(self.sess, './checkpoints/CF_NADE.model', global_step=batch)

    