from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import moving_averages
from .base import BaseLayer
from ..utils import Variable
from .. import init

from tensorflow.python.training import moving_averages
from tensorflow.python.ops import nn
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import tensor_util

__all__ = [
	"BatchNormLayer"
]


class BatchNormLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, is_training, **kwargs):

		# input data shape
		self.incoming = incoming
		incoming_shape = self.incoming.get_output_shape()

		self.bn_axes = [0]
		if len(incoming_shape) > 2:
			self.bn_axes = [0, 1, 2]
		else:
			self.bn_axes = [0]
		
		bn_shape = incoming_shape[-1:]
		self.gamma = Variable(var=init.Constant(value=1.0), shape=bn_shape, regularize=False)
		self.beta = Variable(var=init.Constant(value=0.0), shape=bn_shape, regularize=False)

		self.epsilon = 1e-3
		if 'epsilon' in kwargs.keys():
			self.epsilon = kwargs['epsilon']
		self.decay = 0.99
		if 'momentum' in kwargs.keys():
			self.decay = kwargs['momentum']

		self.is_training = is_training
		self.pop_mean = tf.train.ExponentialMovingAverage(decay=self.decay)
		self.pop_var = tf.train.ExponentialMovingAverage(decay=self.decay)
	
	def get_output(self):
		
		#"""
		batch_mean, batch_var = tf.nn.moments(self.incoming.get_output(), self.bn_axes)
		batch_mean = tf.identity(batch_mean)
		batch_var = tf.identity(batch_var)
		def train_normalization():
			pop_mean_op = self.pop_mean.apply([batch_mean])
			pop_var_op = self.pop_var.apply([batch_var])
			with tf.control_dependencies([pop_mean_op, pop_var_op]):
				return nn.batch_normalization(self.incoming.get_output(),
		                                      batch_mean,
		                                      batch_var,
		                                      self.beta.get_variable(),
		                                      self.gamma.get_variable(),
		                                      self.epsilon)
		def test_normalization():
			return  nn.batch_normalization(self.incoming.get_output(),
	                                      self.pop_mean.average(batch_mean),
	                                      self.pop_var.average(batch_var),
	                                      self.beta.get_variable(),
	                                      self.gamma.get_variable(),
	                                      self.epsilon)

		return tf.cond(self.is_training, train_normalization, test_normalization)
		"""
		def train_normalization():
			batch_mean, batch_var = tf.nn.moments(self.incoming.get_output(), self.bn_axes)
			batch_mean = tf.identity(batch_mean)
			batch_var = tf.identity(batch_var)
			pop_mean_op = self.pop_mean.apply([batch_mean])
			pop_var_op = self.pop_var.apply([batch_var])
			return tf.nn.fused_batch_norm(self.incoming.get_output(), self.beta.get_variable(), 
									self.gamma.get_variable(), mean=None, variance=None, 
									epsilon=0.001, is_training=self.is_training)

		def test_normalization():
			batch_mean, batch_var = tf.nn.moments(self.incoming.get_output(), self.bn_axes)
			batch_mean = tf.identity(batch_mean)
			batch_var = tf.identity(batch_var)
			return tf.nn.fused_batch_norm(self.incoming.get_output(), self.beta.get_variable(), 
									self.gamma.get_variable(), mean=self.pop_mean.average(batch_mean), 
									variance=self.pop_var.average(batch_var), epsilon=0.001, 
									is_training=self.is_training)

		return tf.cond(self.is_training, train_normalization, test_normalization)
		"""

	def get_output_shape(self):
		return self.incoming.get_output_shape()

	def get_variable(self):
		return [self.gamma, self.beta]

	def set_trainable(self, status):
		self.gamma.set_trainable(status)
		self.beta.set_trainable(status)
		
	def is_trainable(self):
		return self.gamma.is_trainable()
		
	def is_l1_regularize(self):
		return self.gamma.is_l1_regularize()  
		
	def is_l2_regularize(self):
		return self.gamma.is_l2_regularize() 



"""
class BatchNormLayer(BaseLayer):
	"1D convolutional layer""

	def __init__(self, incoming, is_training, **kwargs):

		# input data shape
		self.incoming = incoming
		incoming_shape = self.incoming.get_output_shape()

		self.bn_axes = [0]
		if len(incoming_shape) > 2:
			self.bn_axes = [0, 1, 2]
		else:
			self.bn_axes = [0, 1, 2]
		
		bn_shape = incoming_shape[-1:]
		self.gamma = Variable(var=init.Constant(value=1.0), shape=bn_shape, regularize=False)
		self.beta = Variable(var=init.Constant(value=0.0), shape=bn_shape, regularize=False)

		self.epsilon = 1e-3
		if 'epsilon' in kwargs.keys():
			self.epsilon = kwargs['epsilon']
		self.decay = 0.99
		if 'momentum' in kwargs.keys():
			self.decay = kwargs['momentum']

		self.is_training = is_training
		self.pop_mean = init_ops.zeros_initializer()
		self.pop_var = init_ops.ones_initializer()
	
	def get_output(self):
		
		def train_normalization():
			mean, variance = tf.nn.moments(self.incoming.get_output(), self.bn_axes)
			mean_update = moving_averages.assign_moving_average(
							self.pop_mean, mean, self.decay, zero_debias=False)
			variance_update = moving_averages.assign_moving_average(
							self.pop_var, variance, self.decay, zero_debias=False)
			self.updates.append(mean_update)
			self.updates.append(variance_update)
			return nn.batch_normalization(self.incoming.get_output(),	                                      tf.identity(batch_mean),
	                                      tf.identity(mean),
	                                      tf.identity(var),
	                                      self.beta.get_variable(),
	                                      self.gamma.get_variable(),
	                                      self.epsilon)

		def test_normalization():
			return  nn.batch_normalization(self.incoming.get_output(),
	                                      self.pop_mean,
	                                      self.pop_var,
	                                      self.beta.get_variable(),
	                                      self.gamma.get_variable(),
	                                      self.epsilon)

		return control_flow_ops.cond(self.is_training, train_normalization, test_normalization)

	def get_output_shape(self):
		return self.incoming.get_output_shape()

	def get_variable(self):
		return [self.gamma, self.beta]

	def set_trainable(self, status):
		self.gamma.set_trainable(status)
		self.beta.set_trainable(status)
		
	def is_trainable(self):
		return self.gamma.is_trainable()
		
	def is_l1_regularize(self):
		return self.gamma.is_l1_regularize()  
		
	def is_l2_regularize(self):
		return self.gamma.is_l2_regularize() 







	def get_output(self):
		batch_mean, batch_var = tf.nn.moments(self.incoming.get_output(), self.bn_axes)
		def update_mean_var():
			pop_mean_op = self.pop_mean.apply([batch_mean])
			pop_var_op = self.pop_var.apply([batch_var])
			with tf.control_dependencies([pop_mean_op, pop_var_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		def population_mean_var():
			return self.pop_mean.average(batch_mean), self.pop_var.average(batch_var)

		mean, var = tf.cond(self.is_training, update_mean_var, 
											  population_mean_var)

		return tf.nn.batch_normalization(self.incoming.get_output(), mean, var, 
		                                 self.beta.get_variable(), self.gamma.get_variable(), self.epsilon)



	def get_output(self):
		
		mean, variance = tf.nn.moments(self.incoming.get_output(), self.bn_axes)
		if not self.updates
			update_moving_mean = moving_averages.assign_moving_average(
						self.moving_mean, mean, self.decay, zero_debias=False)
			update_moving_variance = moving_averages.assign_moving_average(
						self.moving_variance, variance, self.decay, zero_debias=False)
		self.updates.append(mean_update)
		self.updates.append(variance_update)

		def normalize_in_test():
			return  nn.batch_normalization(inputs,
	                                      self.moving_mean,
	                                      self.moving_variance,
	                                      self.beta,
	                                      self.gamma,
	                                      self.epsilon)

		def normalize_in_training():
			return nn.batch_normalization(inputs,
	                                      mean,
	                                      variance,
	                                      self.beta,
	                                      self.gamma,
	                                      self.epsilon)

		return control_flow_ops.cond(self.is_training, normalize_in_test, normalize_in_training)


"""




