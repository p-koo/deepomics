from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import tensorflow as tf
from .base import BaseLayer
from ..utils import Variable
from .. import init

__all__ = [
	"Conv1DResidualLayer", 
	"Conv2DResidualLayer",
	"DenseResidualLayer"
]





class Conv1DResidualLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, survival_rate, is_training, activation='relu', **kwargs):

		self.survival_rate = self.survival_rate
		self.filter_size = filter_size
		self.is_training = is_training
		self.activation = activation
		self.incoming_shape = incoming.get_output_shape()
		self.num_filters = self.incoming_shape[-1].value


		if 'name' in kwargs:
			self.name = kwargs['name'] + '_'
		else:
			self.name = ''

		if 'keep_prob' in kwargs:
			self.keep_prob = keep_prob
		else:
			self.keep_prob=None

		self.outgoing, self.name = self.conv1d_residual_block()
			
		# shape of the output
		self.output_shape = self.incoming_shape


	def conv1d_residual_block(self):

		# original residual unit
		shape = self.incoming.get_output_shape()
		num_filters = shape[-1].value

		self.outgoing = OrderedDict()
		self.outgoing[self.name+'1resid'] = layers.Conv1DLayer(incoming, num_filters=self.num_filters, filter_size=self.filter_size, padding='SAME')
		self.outgoing[self.name+'1resid_norm'] = layers.BatchNormLayer(self.outgoing[self.name+'1resid'], self.is_training)
		self.outgoing[self.name+'1resid_active'] = layers.ActivationLayer(self.outgoing[self.name+'1resid_norm'], function=self.activation)

		if 'dropout_block' in model_layer:
			self.outgoing[self.name+'dropout1'] = layers.DropoutLayer(self.outgoing[self.name+'1resid_active'], keep_prob=self.keep_prob)
			lastname = self.name+'dropout1'
		else:
			lastname = self.name+'1resid_active'

		self.outgoing[self.name+'2resid'] = layers.Conv1DLayer(self.outgoing[lastname], num_filters=self.num_filters, filter_size=self.filter_size, padding='SAME')
		self.outgoing[self.name+'2resid_norm'] = layers.BatchNormLayer(self.outgoing[self.name+'2resid'], self.is_training)
	
	
	def get_output(self):

			def not_dropped():
					add = tf.add(self.incoming.get_output(), self.outgoing[self.name+'resid'].get_output())
					return tf.nn.relu(add)

			def dropped():
					return tf.nn.relu(self.incoming.get_output())

			def train():
					Z = tf.random_uniform(shape=[], minval=0.0, maxval=1.0, name='survival')
					survive = tf.less(Z, self.survival_rate)
					return tf.cond(survive, not_dropped, dropped)

			def test():
					mul = tf.mul(outgoing.get_output(), self.survival_rate)
					add = tf.add(res, mul)
					return tf.nn.relu(add)

			return tf.cond(self.is_training, train, test)

	def get_input_shape(self):
		return self.incoming_shape


	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):

		params = []
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'is_trainable'):
				if self.outgoing[layer].is_trainable():
					variables = self.outgoing[layer].get_variable()
					if isinstance(variables, list):
						for var in variables:
							params.append(var.get_variable())
					else:
						params.append(variables.get_variable())
		return params
	
	def set_trainable(self, status):
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'set_trainable'):
				self.outgoing[layer].set_trainable(status)

	def set_l1_regularize(self, status):
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'set_l1_regularize'):
				self.outgoing[layer].set_l1_regularize(status)
		
	def set_l2_regularize(self, status):
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'set_l2_regularize'):
				self.outgoing[layer].set_l2_regularize(status)
		
	def is_trainable(self):
		return self.outgoing[self.name+'1resid'].is_trainable()
		
	def is_l1_regularize(self):
		return self.outgoing[self.name+'1resid'].is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.outgoing[self.name+'1resid'].is_l2_regularize()  
		




class Conv2DResidualLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, survival_rate, is_training, activation='relu', **kwargs):

		self.survival_rate = self.survival_rate
		self.filter_size = filter_size
		self.is_training = is_training
		self.activation = activation

		self.incoming_shape = incoming.get_output_shape()
		self.num_filters = self.incoming_shape[-1].value


		if 'name' in kwargs:
			self.name = kwargs['name'] + '_'
		else:
			self.name = ''

		if 'keep_prob' in kwargs:
			self.keep_prob = keep_prob
		else:
			self.keep_prob=None

		self.outgoing, self.name = self.conv2d_residual_block()

			
		# shape of the output
		self.output_shape = self.incoming_shape


	def conv2d_residual_block(self):

		# original residual unit
		shape = self.incoming.get_output_shape()
		num_filters = shape[-1].value

		self.outgoing = OrderedDict()
		self.outgoing[self.name+'1resid'] = layers.Conv2DLayer(incoming, num_filters=self.num_filters, filter_size=self.filter_size, padding='SAME')
		self.outgoing[self.name+'1resid_norm'] = layers.BatchNormLayer(self.outgoing[self.name+'1resid'], self.is_training)
		self.outgoing[self.name+'1resid_active'] = layers.ActivationLayer(self.outgoing[self.name+'1resid_norm'], function=self.activation)

		if 'dropout_block' in model_layer:
			self.outgoing[self.name+'dropout1'] = layers.DropoutLayer(self.outgoing[self.name+'1resid_active'], keep_prob=self.keep_prob)
			lastname = self.name+'dropout1'
		else:
			lastname = self.name+'1resid_active'

		self.outgoing[self.name+'2resid'] = layers.Conv2DLayer(self.outgoing[lastname], num_filters=self.num_filters, filter_size=self.filter_size, padding='SAME')
		self.outgoing[self.name+'2resid_norm'] = layers.BatchNormLayer(self.outgoing[self.name+'2resid'], self.is_training)
	
	
	def get_output(self):

			def not_dropped():
					add = tf.add(self.incoming.get_output(), self.outgoing[self.name+'resid'].get_output())
					return tf.nn.relu(add)

			def dropped():
					return tf.nn.relu(self.incoming.get_output())

			def train():
					Z = tf.random_uniform(shape=[], minval=0.0, maxval=1.0, name='survival')
					survive = tf.less(Z, self.survival_rate)
					return tf.cond(survive, not_dropped, dropped)

			def test():
					mul = tf.mul(outgoing.get_output(), self.survival_rate)
					add = tf.add(res, mul)
					return tf.nn.relu(add)

			return tf.cond(self.is_training, train, test)

	def get_input_shape(self):
		return self.incoming_shape


	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):

		params = []
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'is_trainable'):
				if self.outgoing[layer].is_trainable():
					variables = self.outgoing[layer].get_variable()
					if isinstance(variables, list):
						for var in variables:
							params.append(var.get_variable())
					else:
						params.append(variables.get_variable())
		return params
	
	def set_trainable(self, status):
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'set_trainable'):
				self.outgoing[layer].set_trainable(status)

	def set_l1_regularize(self, status):
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'set_l1_regularize'):
				self.outgoing[layer].set_l1_regularize(status)
		
	def set_l2_regularize(self, status):
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'set_l2_regularize'):
				self.outgoing[layer].set_l2_regularize(status)
		
	def is_trainable(self):
		return self.outgoing[self.name+'1resid'].is_trainable()
		
	def is_l1_regularize(self):
		return self.outgoing[self.name+'1resid'].is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.outgoing[self.name+'1resid'].is_l2_regularize()  
		


class DenseResidualLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, survival_rate, is_training, activation='relu', **kwargs):

		self.survival_rate = self.survival_rate
		self.is_training = is_training
		self.activation = activation

		if 'name' in kwargs:
			self.name = kwargs['name'] + '_'
		else:
			self.name = ''

		if 'keep_prob' in kwargs:
			self.keep_prob = keep_prob
		else:
			self.keep_prob=None

		# input data shape
		self.incoming_shape = incoming.get_output_shape()
		self.num_units = self.incoming_shape[-1].value

		self.outgoing, self.name = self.dense_residual_block()

		# shape of the output
		self.output_shape = self.incoming_shape


	def dense_residual_block(self):

		self.outgoing = OrderedDict()
		self.outgoing[self.name+'1resid'] = layers.DenseLayer(incoming, num_units=self.num_units, b=None)
		self.outgoing[self.name+'1resid_norm'] = layers.BatchNormLayer(self.outgoing[self.name+'1resid'], self.is_training)
		self.outgoing[self.name+'1resid_active'] = layers.ActivationLayer(self.outgoing[self.name+'1resid_norm'], function=self.activation)

		if 'dropout_block' in model_layer:
			self.outgoing[self.name+'dropout1'] = layers.DropoutLayer(self.outgoing[self.name+'1resid_active'], keep_prob=self.keep_prob)
			lastname = self.name+'dropout1'
		else:
			lastname = self.name+'1resid_active'

		self.outgoing[self.name+'2resid'] = layers.DenseLayer(self.outgoing[lastname], num_units=self.num_units, b=None)
		self.outgoing[self.name+'2resid_norm'] = layers.BatchNormLayer(self.outgoing[self.name+'2resid'], self.is_training)
	
	
	def get_output(self):

			def not_dropped():
					add = tf.add(self.incoming.get_output(), self.outgoing[self.name+'resid'].get_output())
					return tf.nn.relu(add)

			def dropped():
					return tf.nn.relu(self.incoming.get_output())

			def train():
					Z = tf.random_uniform(shape=[], minval=0.0, maxval=1.0, name='survival')
					survive = tf.less(Z, self.survival_rate)
					return tf.cond(survive, not_dropped, dropped)

			def test():
					mul = tf.mul(outgoing.get_output(), self.survival_rate)
					add = tf.add(res, mul)
					return tf.nn.relu(add)

			return tf.cond(self.is_training, train, test)

	def get_input_shape(self):
		return self.incoming_shape


	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):

		params = []
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'is_trainable'):
				if self.outgoing[layer].is_trainable():
					variables = self.outgoing[layer].get_variable()
					if isinstance(variables, list):
						for var in variables:
							params.append(var.get_variable())
					else:
						params.append(variables.get_variable())
		return params
	
	def set_trainable(self, status):
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'set_trainable'):
				self.outgoing[layer].set_trainable(status)

	def set_l1_regularize(self, status):
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'set_l1_regularize'):
				self.outgoing[layer].set_l1_regularize(status)
		
	def set_l2_regularize(self, status):
		for layer in self.outgoing:
			if hasattr(self.outgoing[layer], 'set_l2_regularize'):
				self.outgoing[layer].set_l2_regularize(status)
		
	def is_trainable(self):
		return self.outgoing[self.name+'1resid'].is_trainable()
		
	def is_l1_regularize(self):
		return self.outgoing[self.name+'1resid'].is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.outgoing[self.name+'1resid'].is_l2_regularize()  
		