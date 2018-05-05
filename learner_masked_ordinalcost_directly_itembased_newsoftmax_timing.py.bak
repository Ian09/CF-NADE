'''
Created on Nov 25, 2015

@author: yin.zheng
'''


import theano 
import theano.tensor as T
import numpy as np
from blocks.bricks import Rectifier, Softmax, Identity, NDimensionalSoftmax, Tanh, Logistic, Softplus
from blocks.initialization import Constant, Uniform
from blocks.bricks import Initializable, Sequence, Feedforward, Linear, Brick
from blocks.bricks import shared_floatx_nans, add_role
from blocks.roles import WEIGHT, BIAS
from blocks.bricks.base import application
from toolz import interleave
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.wrappers import WithExtraDims
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.algorithms import GradientDescent, Scale
from fuel.datasets import H5PYDataset
from fuel.schemes import (SequentialScheme, ShuffledScheme, SequentialExampleScheme, ShuffledExampleScheme)
from fuel.transformers import ForceFloatX
from fuel.streams import DataStream
from hulu_fuel.datasets.MovieLens1M import MovieLens1M
from fuel.transformers import Transformer
from blocks.extensions import Timing
# from mercurial.revset import extra
# from blaze.expr.reductions import std

def masked_softmax_entropy(h, output_masks, masks):
    h -= h.max(axis=1, keepdims=True)
    logp = (h - T.log((T.exp(h) * masks).sum(axis=1, keepdims=True))) * masks
    return -(output_masks * logp)


def convert_onehot_to_gaussian(one_hot_ratings, std=1):
    mask = one_hot_ratings.sum(axis=2)
    S = np.array([1, 2, 3, 4, 5], dtype='float32')
    ratings = T.argmax(one_hot_ratings, axis=2) + 1
    scores = ratings.dimshuffle(0, 1, 'x') - S[None, None, :]
    unnormalized_score = T.exp(-(scores ** 2) / (2 * std ** 2))
    gaussian = mask[:, :, None] * unnormalized_score / (T.sum(unnormalized_score, axis=2)[:, :, None])
    return gaussian

def rating_cost(pred_score, true_ratings, input_masks, output_masks, D, d, std=1.0, alpha=0.01):
    
    pred_score_cum = T.extra_ops.cumsum(pred_score, axis=2)
    prob_item_ratings = NDimensionalSoftmax(name='rating_cost_sf').apply(pred_score_cum, extra_ndim=1)
    accu_prob_1N = T.extra_ops.cumsum(prob_item_ratings, axis=2)
    accu_prob_N1 = T.extra_ops.cumsum(prob_item_ratings[:, :, ::-1], axis=2)[:, :, ::-1]
    mask1N = T.extra_ops.cumsum(true_ratings[:, :, ::-1], axis=2)[:, :, ::-1]
    maskN1 = T.extra_ops.cumsum(true_ratings, axis=2)
    cost_ordinal_1N = -T.sum((T.log(prob_item_ratings) - T.log(accu_prob_1N)) * mask1N, axis=2)
    cost_ordinal_N1 = -T.sum((T.log(prob_item_ratings) - T.log(accu_prob_N1)) * maskN1, axis=2)
    cost_ordinal = cost_ordinal_1N + cost_ordinal_N1
    nll_item_ratings = -(true_ratings * T.log(prob_item_ratings)).sum(axis=2)
    nll = std * nll_item_ratings.sum(axis=1) * 1.0 * D / (D - d + 1e-6) + alpha * cost_ordinal.sum(axis=1) * 1.0 * D / (D - d + 1e-6)
    cost = T.mean(nll)
    return cost, nll, nll_item_ratings, cost_ordinal_1N, cost_ordinal_N1, prob_item_ratings

def RMSE(pred_ratings, true_ratings):
    pass

class MovieLensTransformer(Transformer):
    
    def __init__(self, data_stream, seed=1234):
        super(MovieLensTransformer, self).__init__(data_stream)
        self.data_sources = ('input_ratings',
                             'output_ratings',
                             'input_masks',
                             'output_masks')
        self.produces_examples = False
    @property
    def sources(self):
        return self.data_sources
    
    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        batch = next(self.child_epoch_iterator)
        inp_ratings, out_ratings, input_masks, output_masks = self.preprocess_data(batch)
    
        return inp_ratings, out_ratings, input_masks, output_masks
    
    def preprocess_data(self, batch):
        
        input_ratings, output_ratings, input_masks, output_masks = batch
        input_shape = input_ratings.shape
        K = 5
        input_ratings_3d = np.zeros((input_shape[0], input_shape[1], K), 'int8')
        output_ratings_3d = np.zeros_like(input_ratings_3d)
        input_ratings_nonzero = input_ratings.nonzero()
        input_ratings_3d[input_ratings_nonzero[0],
                         input_ratings_nonzero[1],
                         input_ratings[input_ratings_nonzero[0],
                                       input_ratings_nonzero[1]
                                       ] - 1] = 1
        output_ratings_nonzero = output_ratings.nonzero()
        output_ratings_3d[output_ratings_nonzero[0],
                         output_ratings_nonzero[1],
                         output_ratings[output_ratings_nonzero[0],
                                       output_ratings_nonzero[1]
                                       ] - 1] = 1
                                       
        
        
        return (input_ratings_3d, output_ratings_3d, input_masks, output_masks)

class Trainer_MovieLensTransformer(Transformer):
    
    def __init__(self, data_stream, seed=1234):
        super(Trainer_MovieLensTransformer, self).__init__(data_stream)
        self.data_sources = ('input_ratings',
                             'output_ratings',
                             'input_masks',
                             'output_masks')
        self.produces_examples = False
        
        
    @property
    def sources(self):
        return self.data_sources
    
    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        batch = next(self.child_epoch_iterator)
        inp_ratings, out_ratings, input_masks, output_masks = self.preprocess_data(batch)
    
        return inp_ratings, out_ratings, input_masks, output_masks
    
    def preprocess_data(self, batch):
        
        ratings, _, _, _ = batch
#         valid_ratings = np.array(ratings > 0, 'int8')
        input_masks = np.zeros_like(ratings)
        output_masks = np.zeros_like(ratings)
        input_ratings = np.zeros_like(ratings)
        output_ratings = np.zeros_like(ratings)
        
        cnt = 0
        for rat in ratings:
            nonzero_id = rat.nonzero()[0]
            if len(nonzero_id) == 0:
                continue
            ordering = np.random.permutation(np.arange(len(nonzero_id)))
            d = np.random.randint(0, len(ordering))
            flag_in = (ordering < d)
            flag_out = (ordering >= d)
            input_masks[cnt][nonzero_id] = flag_in
            output_masks[cnt][nonzero_id] = flag_out
            input_ratings[cnt] = rat * input_masks[cnt]
            output_ratings[cnt] = rat * output_masks[cnt]
            cnt += 1
        return (input_ratings, output_ratings, input_masks, output_masks)
            
def get_done_text(start_time):
    sys.stdout.flush()
    return "DONE in {:.4f} seconds.\n".format(t.time() - start_time)
            
        
        

    

class TensorLinear(Initializable):
    def __init__(self, input_dim0, input_dim1, output_dim,
                 batch_size, **kwargs):
        
        super(TensorLinear, self).__init__(**kwargs)
        self.input_dim0 = input_dim0
        self.input_dim1 = input_dim1
        self.output_dim = output_dim
        
    def __allocate(self, input_dim0, input_dim1, output_dim):
        W = shared_floatx_nans((input_dim0, input_dim1, output_dim), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        b = shared_floatx_nans((output_dim,), name='b')
        add_role(b, BIAS)
        self.parameters.append(b)
        Q = shared_floatx_nans((input_dim0,output_dim),name='Q')
        add_role(Q, WEIGHT)
        self.parameters.append(Q)
        
        
    
    def _allocate(self):
        self.__allocate(self.input_dim0, self.input_dim1, self.output_dim)
        
    def _initialize(self):
        W, b, Q = self.parameters
        self.weights_init.initialize(W, self.rng)
        self.biases_init.initialize(b, self.rng)
        self.weights_init.initialize(Q, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        W, b, Q = self.parameters
#         input_ = input_ / (T.sum(input_, axis=(1,2))[:, None, None]+1e-6)
        output_ = T.tensordot(input_, W, axes=[[1, 2], [0, 1]]) + b
        input_mask = T.sum(input_, axis = 2)
        output_masked = T.dot(input_mask, Q) 
        output = output_ + output_masked
#         output = output / (T.sum(input_, axis=(1,2))[:,None] + 1)
        return output
    
    def get_dim(self, name):
        if name == 'input_':
            return (self.input_dim0, self.input_dim1)
        if name == 'output':
            return self.output_dim
        super(TensorLinear, self).get_dim(name)

class TensorLinear_inverse(Initializable):
    def __init__(self, input_dim, output_dim0, output_dim1,
                 batch_size, **kwargs):
        
        super(TensorLinear_inverse, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim0 = output_dim0
        self.output_dim1 = output_dim1
        
    def __allocate(self, input_dim, output_dim0, output_dim1):
        W = shared_floatx_nans((input_dim, output_dim0, output_dim1), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        b = shared_floatx_nans((output_dim0, output_dim1), name='b')
        add_role(b, BIAS)
        self.parameters.append(b)
        
    
    def _allocate(self):
        self.__allocate(self.input_dim, self.output_dim0, self.output_dim1)
        
    def _initialize(self):
        W, b = self.parameters
        self.weights_init.initialize(W, self.rng)
        self.biases_init.initialize(b, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        W, b = self.parameters
        output = T.tensordot(input_, W, axes=[[1], [0]]) + b
        return output
    
    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return (self.output_dim0, self.output_dim1)
        super(TensorLinear_inverse, self).get_dim(name)
        
class TensorLinear_Plus_Linear(Initializable):
    def __init__(self, input_dim0, input_dim1, output_dim,
                 batch_size, **kwargs):
        '''
        input_dim0: number of items
        input_dim1: number of ratings (1~input_dim1), a.k.a K in our paper
        '''
        super(TensorLinear_Plus_Linear, self).__init__(**kwargs)
        self.input_dim0 = input_dim0
        self.input_dim1 = input_dim1
        self.output_dim = output_dim
        
    def __allocate(self, input_dim0, input_dim1, output_dim):
        W = shared_floatx_nans((input_dim0, input_dim1, output_dim), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        b = shared_floatx_nans((output_dim,), name='b')
        add_role(b, BIAS)
        self.parameters.append(b)
        W_Linear = shared_floatx_nans((input_dim0, output_dim), name='W_Linear')
        self.add_auxiliary_variable(W_Linear.norm(2), name='W_Linear_norm')
        add_role(W_Linear, WEIGHT)
        self.parameters.append(W_Linear)
        
    
    def _allocate(self):
        self.__allocate(self.input_dim0, self.input_dim1, self.output_dim)
        
    def _initialize(self):
        W, b, W_Linear = self.parameters
        self.weights_init.initialize(W, self.rng)
        self.biases_init.initialize(b, self.rng)
        self.weights_init.initialize(W_Linear, self.rng)

    @application(inputs=['input0_', 'input1_'], outputs=['output'])
    def apply(self, input0_, input1_):
        W, b, W_Linear = self.parameters
        output = T.tensordot(input0_, W, axes=[[1, 2], [0, 1]]) + T.dot(input1_, W_Linear) + b
        return output
    
    def get_dim(self, name):
        if name == 'input_':
            return (self.input_dim0, self.input_dim1)
        if name == 'output':
            return self.output_dim
        super(TensorLinear_Plus_Linear, self).get_dim(name)


def Adam_optimizer(input_list, cost, parameters, lr0, b1, b2, epsilon):
    
    params_gradient = [T.grad(cost, param) for param in parameters]
    grad_shared = [theano.shared(p.get_value() * 0., name='%s_grad' % p.name) for p in parameters]
    grads_update = [(gs, g) for gs, g in zip(grad_shared, params_gradient)]
    f_get_grad = theano.function(inputs=input_list,
                                  updates=grads_update,
                                  outputs=cost,
                                  )
    
    updates = []
    
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1 ** (i_t)
    fix2 = 1. - b2 ** (i_t)
    lr_t = lr0 * (T.sqrt(fix2) / fix1)

    for p, g in zip(parameters, grad_shared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update_parameters = theano.function([lr0], [], updates=updates)
    
    return f_get_grad, f_update_parameters, grad_shared

def Adadelta_optimizer(input_list, cost, parameters, decay, epsilon):
    
    params_gradient = [T.grad(cost, param) for param in parameters]
    grad_shared = [theano.shared(p.get_value() * 0., name='%s_grad' % p.name) for p in parameters]
    running_up2 = [theano.shared(p.get_value() * 0, name='%s_rup2' % p.name) for p in parameters]
    running_grads2 = [theano.shared(p.get_value() * 0, name='%s_rgrad2' % p.name) for p in parameters]
    zgup = [(zg, g) for zg, g in zip(grad_shared, params_gradient)]
    rg2up = [(rg2, decay * rg2 + (1.0 - decay) * (g ** 2)) for rg2, g in zip(running_grads2, params_gradient)]
    
    f_get_grad = theano.function(inputs=input_list,
                                  updates=zgup + rg2up,
                                  outputs=cost,
                                  )
    
    updir = [-T.sqrt(ru2 + epsilon) / T.sqrt(rg2 + epsilon) * zg for zg, ru2, rg2 in zip(grad_shared, running_up2, running_grads2)]
    ru2up = [(ru2, decay * ru2 + (1 - decay) * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(parameters, updir)]
    
    

    f_update_parameters = theano.function([], [], updates=ru2up + param_up)
    
    return f_get_grad, f_update_parameters, grad_shared
    
def SGD_optimizer(input_list, cost, parameters, lr0, mu):
    
    params_gradient = [T.grad(cost, param) for param in parameters]
    grad_shared = [theano.shared(p.get_value() * 0., name='%s_grad' % p.name) for p in parameters]
    velo_shared = [theano.shared(p.get_value() * 0., name='%s_velocity' % p.name) for p in parameters]
    
    grads_update = [(gs, g) for gs, g in zip(grad_shared, params_gradient)]
    f_get_grad = theano.function(inputs=input_list,
                                  updates=grads_update,
                                  outputs=cost,
                                  )
    
    updates = []
    for p, v, g in zip(parameters, velo_shared, grad_shared):
        
        p_t = p - lr0 * g + mu * v
        v_t = mu * v - lr0 * g
        updates.append((p, p_t))
        updates.append((v, v_t))

    f_update_parameters = theano.function([lr0], [], updates=updates)
    
    return f_get_grad, f_update_parameters, grad_shared  

def polyak(parameters, mu):
    
    polyak_shared = [theano.shared(p.get_value(), name='%s_polyak' % p.name) for p in parameters]
    updates = []
    for y, p in zip(polyak_shared, parameters):
        y_t = mu * y + (1 - mu) * p
        updates.append((y, y_t))
    f_update_polyak = theano.function([], [], updates=updates)
    
    return f_update_polyak, polyak_shared

def polyak_replace(parameters, polyaks):
    
    updates = []
    for y, p in zip(polyaks, parameters):
        y_name_split = y.name.split('_')
        
        assert y_name_split[0] == p.name
        updates.append((p, y))
    
    f_replace_polyak = theano.function([], [], updates=updates)
    return f_replace_polyak
        

class tabula_NADE(Sequence, Initializable, Feedforward):
    
    def __init__(self, input_dim0, input_dim1, other_dims, activations, batch_size,
                 **kwargs):
        
        self.activations = activations
        self.input_dim0 = input_dim0
        self.input_dim1 = input_dim1
        self.other_dims = other_dims
        self.batch_size = batch_size
        self.linear_transformations = []
        self.linear_transformations.append(TensorLinear(input_dim0=self.input_dim0,
                                                        input_dim1=self.input_dim1,
                                                        output_dim=self.other_dims[0],
                                                        batch_size=batch_size)
                                           )
        self.linear_transformations.extend([Linear(name='linear_{}'.format(i),
                                                   input_dim=other_dims[i],
                                                   output_dim=other_dims[i + 1])
                                            for i in range(len(other_dims) - 1)])
        self.linear_transformations.append(TensorLinear_inverse(input_dim=self.other_dims[-1],
                                                                output_dim0=self.input_dim0,
                                                                output_dim1=self.input_dim1,
                                                                batch_size=batch_size))
        application_methods = []
        for entity in interleave([self.linear_transformations, activations]):
            if entity is None:
                continue
            if isinstance(entity, Brick):
                application_methods.append(entity.apply)
            else:
                application_methods.append(entity)
        super(tabula_NADE, self).__init__(application_methods, **kwargs)
        

        
    @property
    def input_dim(self):
        return (self.input_dim0, input_dim1)

    @input_dim.setter
    def input_dim(self, value):
        self.input_dim0 = value[0]
        self.input_dim1 = value[1]

    @property
    def hidden_dims(self):
        return self.other_dims

    @hidden_dims.setter
    def hidden_dims(self, value):
        self.other_dims = value
    
    
if __name__ == '__main__':
    
    import sys, os
    import time as t
    import copy as cp
    import gc
    from blocks.graph import ComputationGraph, apply_dropout
    from blocks.filter import VariableFilter
    from blocks.roles import INPUT, OUTPUT
    sys.argv.pop(0)
    
    
    input_dim0 = 6040
    input_dim1 = 5
#     output_dim = 128
    batch_size = int(sys.argv[0])
    n_iter = int(sys.argv[1])
    look_ahead = int(sys.argv[2])
    lr = float(sys.argv[3])  # lr in Adam and SGD, decay in Adadelta
    b1 = float(sys.argv[4])  # b1 in Adam, mu in SGD
    b2 = float(sys.argv[5])
    epsilon = float(sys.argv[6])
    hidden_size_split = (sys.argv[7]).split('_')
    hidden_size = [int(x) for x in hidden_size_split]
    activation_function = sys.argv[8]
    drop_rate = float(sys.argv[9])
    weight_decay = float(sys.argv[10])
    Optimizer = sys.argv[11]
    std = float(sys.argv[12])
    alpha = float(sys.argv[13])
    polyak_mu = float(sys.argv[14])
    output_path = sys.argv[15]
    
    
    np.random.seed(12345)
    
    
    
#     input_mat = np.random.rand(batch_size, input_dim0, input_dim1).astype(theano.config.floatX)
#     input_ratings = np.zeros((batch_size, input_dim0, input_dim1), dtype=theano.config.floatX)
#     input_ratings[:, :, 0] = 1
    trainset = MovieLens1M(which_set=['train'], sources=('input_ratings', 'output_ratings', 'input_masks', 'output_masks'))
    validset = MovieLens1M(which_set=['valid'], sources=('input_ratings', 'output_ratings', 'input_masks', 'output_masks'))
    testset = MovieLens1M(which_set=['test'], sources=('input_ratings', 'output_ratings', 'input_masks', 'output_masks'))
    
    train_loop_stream = ForceFloatX(
                                    data_stream=MovieLensTransformer(
                                                                     
                                            data_stream=Trainer_MovieLensTransformer(
                                                                                     data_stream=DataStream(
                                                                                                            dataset=trainset,
                                                                                                            iteration_scheme=ShuffledScheme(
                                                                                                                                            trainset.num_examples,
                                                                                                                                            batch_size
                                                                                                                                            )
                                                                                                            )
                                                                                     )
                                                                     )
                                    )
    
    trainval_loop_stream = ForceFloatX(
                                    data_stream=MovieLensTransformer(
                                                                     
                                            data_stream=Trainer_MovieLensTransformer(
                                                                                     data_stream=DataStream(
                                                                                                            dataset=testset,
                                                                                                            iteration_scheme=ShuffledScheme(
                                                                                                                                            trainset.num_examples,
                                                                                                                                            batch_size
                                                                                                                                            )
                                                                                                            )
                                                                                     )
                                                                     )
                                    )
    
#     train_monitor_stream = ForceFloatX(
#                             data_stream=MovieLensTransformer(
#                                                     data_stream=DataStream(
#                                                                            dataset=trainset,
#                                                                            iteration_scheme=ShuffledScheme(
#                                                                                                            validset.num_examples,
#                                                                                                            batch_size
#                                                                                                            )
#                                                                            
#                                                                            )
#                                                     
#                                                     )
#                                        )
    
    valid_monitor_stream = ForceFloatX(
                            data_stream=MovieLensTransformer(
                                                    data_stream=DataStream(
                                                                           dataset=validset,
                                                                           iteration_scheme=ShuffledScheme(
                                                                                                           validset.num_examples,
                                                                                                           batch_size
                                                                                                           )
                                                                           
                                                                           )
                                                    
                                                    )
                                       )
    
    test_monitor_stream = ForceFloatX(
                            data_stream=MovieLensTransformer(
                                                    data_stream=DataStream(
                                                                           dataset=testset,
                                                                           iteration_scheme=ShuffledScheme(
                                                                                                           testset.num_examples,
                                                                                                           batch_size
                                                                                                           )
                                                                           
                                                                           )
                                                    
                                                    )
                                       )
    
    rating_freq = np.zeros((6040, 5))
    init_b = np.zeros((6040, 5))
    for batch in valid_monitor_stream.get_epoch_iterator():
        inp_r, out_r, inp_m, out_m = batch
        rating_freq += inp_r.sum(axis=0)
    
    log_rating_freq = np.log(rating_freq + 1e-8)
    log_rating_freq_diff = np.diff(log_rating_freq, axis=1)
    init_b[:, 1:] = log_rating_freq_diff
    init_b[:, 0] = log_rating_freq[:, 0]
#     init_b = np.log(rating_freq / (rating_freq.sum(axis=1)[:, None] + 1e-8) +1e-8)  * (rating_freq>0)
    
    new_items = np.where(rating_freq.sum(axis=1) == 0)[0]
    
    input_ratings = T.tensor3(name='input_ratings', dtype=theano.config.floatX)
    output_ratings = T.tensor3(name='output_ratings', dtype=theano.config.floatX)
    input_masks = T.matrix(name='input_masks', dtype=theano.config.floatX)
    output_masks = T.matrix(name='output_masks', dtype=theano.config.floatX)
    
    input_ratings_cum = T.extra_ops.cumsum(input_ratings[:, :, ::-1], axis=2)[:, :, ::-1]
    
    
#     hidden_size = [256]
    if activation_function == 'reclin':
        act = Rectifier
    elif activation_function == 'tanh':
        act = Tanh
    elif activation_function == 'sigmoid':
        act = Logistic
    elif activation_function == 'softplus':
        act = Softplus
    layers_act = [act('layer_%d' % i) for i in range(len(hidden_size))]
    NADE_CF_model = tabula_NADE(activations=layers_act,
                                input_dim0=input_dim0,
                                input_dim1=input_dim1,
                                other_dims=hidden_size,
                                batch_size=batch_size,
                                weights_init=Uniform(std=0.05),
                                biases_init=Constant(0.0)
                                )
    NADE_CF_model.push_initialization_config()
    dims = [input_dim0] + hidden_size + [input_dim0]
    linear_layers = [layer for layer in NADE_CF_model.children
                     if 'linear' in layer.name]
    assert len(linear_layers) == len(dims) - 1
    for i in range(len(linear_layers)):
        H1 = dims[i]
        H2 = dims[i + 1]
        width = 2 * np.sqrt(6) / np.sqrt(H1 + H2)
#         std = np.sqrt(2. / dim)
        linear_layers[i].weights_init = Uniform(width=width)
    
    
#     NADE_CF_model.children[0].weights_init = Constant(1)
#     NADE_CF_model.children[0].biases_init = Constant(1.5)
#     NADE_CF_model.children[1].weights_init = Constant(2)
#     NADE_CF_model.children[1].biases_init = Constant(2.5)
    NADE_CF_model.initialize()
    NADE_CF_model.children[-1].parameters[-1].set_value(init_b.astype(theano.config.floatX))
    y = NADE_CF_model.apply(input_ratings_cum)
    y_cum = T.extra_ops.cumsum(y, axis=2)
    predicted_ratings = NDimensionalSoftmax().apply(y_cum, extra_ndim=1)
    d = input_masks.sum(axis=1)
    D = (input_masks + output_masks).sum(axis=1)
#     ratings = T.tensor3(name="ratings", dtype=theano.config.floatX)
    cost, nll, nll_item_ratings, cost_ordinal_1N, cost_ordinal_N1, prob_item_ratings = rating_cost(y, output_ratings, input_masks, output_masks, D, d, alpha=alpha, std=std)
    cost.name = 'cost'
    
    cg = ComputationGraph(cost)
    if weight_decay > 0.0:
        all_weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        l2_weights = T.sum([(W ** 2).sum() for W in all_weights])
        l2_cost = cost + weight_decay * l2_weights
        l2_cost.name = 'l2_decay_' + cost.name
        cg = ComputationGraph(l2_cost)
    if drop_rate > 0.0:
        dropped_layer = VariableFilter(roles=[INPUT], bricks=NADE_CF_model.children)(cg.variables)
        dropped_layer = [layer for layer in dropped_layer if 'linear' in layer.name]
        dropped_layer = dropped_layer[1:]
        cg_dropout = apply_dropout(cg, dropped_layer, drop_rate)
    else:
        cg_dropout = cg
    training_cost = cg_dropout.outputs[0]
    lr0 = T.scalar(name='learning_rate', dtype=theano.config.floatX)
    input_list = [input_ratings, input_masks, output_ratings, output_masks]
    if Optimizer == 'Adam':
        f_get_grad, f_update_parameters, shared_gradients = Adam_optimizer(input_list,
                                                                           training_cost,
                                                                           cg_dropout.parameters,
                                                                           lr0,
                                                                           b1,
                                                                           b2,
                                                                           epsilon)
    elif Optimizer == 'Adadelta':
        f_get_grad, f_update_parameters, shared_gradients = Adadelta_optimizer(input_list,
                                                                               training_cost,
                                                                               cg_dropout.parameters,
                                                                               lr,
                                                                               epsilon)
    elif Optimizer == 'SGD':
        f_get_grad, f_update_parameters, shared_gradients = SGD_optimizer(input_list,
                                                                           training_cost,
                                                                           cg_dropout.parameters,
                                                                           lr0,
                                                                           b1)
        
        
#     f_get_grad, f_update_parameters, shared_gradients = SGD_optimizer(input_list,
#                                                                       cost,
#                                                                       cg.parameters,
#                                                                       lr0)
    param_list = []
    [param_list.extend(p.parameters) for p in NADE_CF_model.children]
    f_update_polyak, shared_polyak = polyak(param_list, mu=polyak_mu)
    
#     f_monitor = theano.function(inputs=[input_ratings, input_masks, output_ratings, output_masks],
#                                 outputs=[predicted_ratings,training_cost, nll, nll_item_ratings, cost_ordinal_1N, cost_ordinal_N1, prob_item_ratings])
    f_monitor = theano.function(inputs=[input_ratings],
                                outputs=[predicted_ratings])
    nb_of_epocs_without_improvement = 0
    best_valid_error = np.Inf
    epoch = 0
    test_RMSE = []
    valid_RMSE = []
    test_RMSE = []
    best_model = cp.deepcopy(NADE_CF_model)
    best_polyak = cp.deepcopy(shared_polyak)
    start_training_time = t.time()
    lr_tracer = []
    rate_score = np.array([1, 2, 3, 4, 5], np.float32)
    while(epoch < n_iter and nb_of_epocs_without_improvement < look_ahead):        
        print 'Epoch {0}'.format(epoch)
        epoch += 1
        start_time_epoch = t.time()
        cost_train = []
        squared_error_train = []
        n_sample_train = []
        cntt = 0
        train_time = 0
        for batch in train_loop_stream.get_epoch_iterator():
            
            inp_r, out_r, inp_m, out_m = batch
            train_t = t.time()
            cost_value = f_get_grad(inp_r, inp_m, out_r, out_m)
            train_time += t.time() - train_t
#             pred_ratings = f_monitor(inp_r)
            if Optimizer == 'Adadelta':
                f_update_parameters()
            else:
                f_update_parameters(lr)
            f_update_polyak()
            pred_ratings = f_monitor(inp_r)
            true_r = out_r.argmax(axis=2) + 1
            pred_r = (pred_ratings[0] * rate_score[np.newaxis, np.newaxis, :]).sum(axis=2)
            pred_r[:, new_items] = 3
#             new_users = np.where((inp_m+out_m).sum(axis=1)==0)[0]
#             pred_r[new_users,:] = 3
#             pred_r = pred_ratings[0].argmax(axis=2) + 1
            mask = out_r.sum(axis=2)
            se = np.sum(np.square(true_r - pred_r) * mask)
            n = np.sum(mask)
            squared_error_train.append(se)
            n_sample_train.append(n)
            cost_train.append(cost_value)
            cntt+= 1
            
        cost_train = np.array(cost_train).mean()
        squared_error_ = np.array(squared_error_train).sum()
        n_samples = np.array(n_sample_train).sum()
        train_RMSE = np.sqrt(squared_error_ / (n_samples * 1.0 + 1e-8))
        
        print '\tTraining   ...',
        print 'Train     :', "RMSE: {0:.6f}".format(train_RMSE), " Cost Error: {0:.6f}".format(cost_train), "Train Time: {0:.6f}".format(train_time), get_done_text(start_time_epoch)
        
        print '\tValidating ...',
        start_time = t.time()
        squared_error_valid = []
        n_sample_valid = []
        valid_time = 0
        for batch in valid_monitor_stream.get_epoch_iterator():
            inp_r, out_r, inp_m, out_m = batch
            valid_t = t.time()
            pred_ratings = f_monitor(inp_r)
            valid_time += t.time() - valid_t
            true_r = out_r.argmax(axis=2) + 1
            pred_r = (pred_ratings[0] * rate_score[np.newaxis, np.newaxis, :]).sum(axis=2)
            
            pred_r[:, new_items] = 3
#             new_users = np.where(inp_m.sum(axis=1)==0)[0]
#             pred_r[new_users,:] = 3
            
#             pred_r = pred_ratings[0].argmax(axis=2) + 1
            mask = out_r.sum(axis=2)
            se = np.sum(np.square(true_r - pred_r) * mask)
            n = np.sum(mask)
            squared_error_valid.append(se)
            n_sample_valid.append(n)
        
        squared_error_ = np.array(squared_error_valid).sum()
        n_samples = np.array(n_sample_valid).sum()
        valid_RMSE = np.sqrt(squared_error_ / (n_samples * 1.0 + 1e-8))
        print 'Validation:', " RMSE: {0:.6f}".format(valid_RMSE) , "Valid Time: {0:.6f}".format(valid_time), get_done_text(start_time),
        if valid_RMSE < best_valid_error:
            best_epoch = epoch
            nb_of_epocs_without_improvement = 0
            best_valid_error = valid_RMSE
            del best_model
            del best_polyak
            gc.collect()
            
            best_model = cp.deepcopy(NADE_CF_model)
            best_polyak = cp.deepcopy(shared_polyak)
            print '\n\n Got a good one'
        else:
            nb_of_epocs_without_improvement += 1
            if Optimizer == 'Adadelta':
                pass
            elif nb_of_epocs_without_improvement == look_ahead and lr > 1e-5:
                nb_of_epocs_without_improvement = 0
                lr /= 4 
                print "learning rate is now %s" % lr 
        lr_tracer.append(lr)
                
                
    print '\n### Training, n_layers=%d' % (len(hidden_size)), get_done_text(start_training_time)
    
    best_y = best_model.apply(input_ratings_cum)
    best_y_cum = T.extra_ops.cumsum(best_y, axis=2)
    best_predicted_ratings = NDimensionalSoftmax().apply(best_y_cum, extra_ndim=1)
    f_monitor_best = theano.function(inputs=[input_ratings],
                                outputs=[best_predicted_ratings])
    
    print '\tTesting ...',
    start_time = t.time()
    squared_error_test = []
    n_sample_test = []
    test_time = 0
    for batch in test_monitor_stream.get_epoch_iterator():
        inp_r, out_r, inp_m, _ = batch
        test_t = t.time()
        pred_ratings = f_monitor_best(inp_r)
        test_time += t.time() - test_t
        true_r = out_r.argmax(axis=2) + 1
        pred_r = (pred_ratings[0] * rate_score[np.newaxis, np.newaxis, :]).sum(axis=2)
        pred_r[:, new_items] = 3
#         new_users = np.where(inp_m.sum(axis=1)==0)[0]
#         pred_r[new_users,:] = 3
#         pred_r = pred_ratings[0].argmax(axis=2) + 1
        mask = out_r.sum(axis=2)
        se = np.sum(np.square(true_r - pred_r) * mask)
        n = np.sum(mask)
        squared_error_test.append(se)
        n_sample_test.append(n)
    
    squared_error_ = np.array(squared_error_test).sum()
    n_samples = np.array(n_sample_test).sum()
    test_RMSE = np.sqrt(squared_error_ / (n_samples * 1.0 + 1e-8))
    print 'Test:', " RMSE: {0:.6f}".format(test_RMSE) , "Test Time: {0:.6f}".format(test_time), get_done_text(start_time),
    
    f = open(os.path.join(output_path, 'Reco_NADE_masked_directly_itembased.txt'), 'a')
    to_write = [str(test_RMSE), str(best_valid_error), str(best_epoch)] + sys.argv[:-1]
    line = " ".join(to_write) + '\n'
    f.write(line)
    f.close()
    
    print '\tTesting with polyak parameters...',
    best_param_list = []
    [best_param_list.extend(p.parameters) for p in best_model.children]
    f_replace = polyak_replace(best_param_list, best_polyak)
    f_replace()
    cc = 0
    for pp in best_polyak:
        pp_value = pp.get_value()
        np.save('/tmp/cfnade/%d'%cc, pp_value)
        cc+=1
    
    
    start_time = t.time()
    squared_error_test = []
    n_sample_test = []
    test_time = 0
    for batch in test_monitor_stream.get_epoch_iterator():
        inp_r, out_r, inp_m, _ = batch
        test_t = t.time()
        pred_ratings = f_monitor_best(inp_r)
        test_time += t.time() - test_t
        true_r = out_r.argmax(axis=2) + 1
        pred_r = (pred_ratings[0] * rate_score[np.newaxis, np.newaxis, :]).sum(axis=2)
        pred_r[:, new_items] = 3
#         new_users = np.where(inp_m.sum(axis=1)==0)[0]
#         pred_r[new_users,:] = 3
#         pred_r = pred_ratings[0].argmax(axis=2) + 1
        mask = out_r.sum(axis=2)
        se = np.sum(np.square(true_r - pred_r) * mask)
        n = np.sum(mask)
        squared_error_test.append(se)
        n_sample_test.append(n)
    
    squared_error_ = np.array(squared_error_test).sum()
    n_samples = np.array(n_sample_test).sum()
    test_RMSE = np.sqrt(squared_error_ / (n_samples * 1.0 + 1e-8))
    print 'Test:', " RMSE: {0:.6f}".format(test_RMSE) , "Test Time: {0:.6f}".format(test_time), get_done_text(start_time),
    
    f = open(os.path.join(output_path, 'Reco_NADE_masked_directly_itembased.txt'), 'a')
    to_write = [str(test_RMSE), str(best_valid_error), str(best_epoch)] + sys.argv[:-1] + ['polyak']
    line = " ".join(to_write) + '\n'
    f.write(line)
    f.close()
    
    
