import tensorflow as tf
import numpy as np

class CF_NADE():
	def __init__(self, sess, flags):
		self.sess = sess
		self.X = tf.placeholder("float32",[None, flags.movie_dim, flags.num_classes]) # batch * movie_dim * num_classes
		self.Y = tf.placeholder("float32",[None, 2]) # batch * (ratings, idex)

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
		self.H = tf.tanh(tf.add(bias['b_hidden'],tf.tensordot(self.X,weights['W'], axes=[[1,2],[0,1]])))

		self.output_layer = tf.add(bias['b_output'], tf.tensordot(self.H, weights['Output_W'], axes=[[1],[0]]))
	    #dim(output_layer) = batch_size * num_classes
		self.output_layer = [self.output_layer[i,self.Y[i][1],:] for i in range(flags.batch_size)]

	    #dim(scores_matrix) = batch_size * num_classes
		scores_matrix = np.repeat([[1],[2],[3],[4],[5]], flags.batch_size, axis=0)
		self.pred_scores = tf.reduce_sum(scores_tensor * tf.nn.softmax(self.output_layer, axis=1), axis=1)
		self.true_scores = self.Y[:, 0]
		self.loss_op = tf.losses.mean_squared_error(self.true_scores, self.pred_scores)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=flags.learning_rate)
		self.train_op = optimizer.minimize(self.loss_op)

		self.init = tf.global_variables_initializer()

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

	def train(self, myData, flags):
		self.buildGraph(flags.batch_size)

		self.sess.run(self.init)

		X, Y = myData.get_batch_new(512, 'train')
		X = data_transform(X, flags)

		self.sess.run([self.train_op], feed_dict={self.X:X, self.Y:Y})


    