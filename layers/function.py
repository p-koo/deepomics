import tensorflow as tf
from .base import BaseLayer
from ..utils import Variable
from .. import init


__all__ = [
	"ActivationLayer",
	"BiasLayer",
]


class ActivationLayer(BaseLayer):
	"""activation layer"""

	def __init__(self, incoming, function=[], **kwargs):
		
		self.incoming = incoming
		self.function = function
		if not self.function:
			self.function = 'relu'
			
	def output(self):
		return activation(z=self.incoming.output(), 
								function=self.function)
	


class BiasLayer(BaseLayer):
	"""Bias layer"""
	
	def __init__(self, incoming, b=[], **kwargs):
		
		self.incoming = incoming
		incoming_shape = incoming.output().get_shape()
		if len(incoming_shape) > 2:
			num_units = incoming_shape[3].value
		else:
			num_units = incoming_shape[1].value

			
		if not b:
			self.b = Variable(var=init.Constant(0.05), 
						 	  shape=[num_units], 
							  **kwargs)
		else:
			self.b = Variable(var=b, shape=[num_units], **kwargs)
			
		
	def output(self):
		return  self.incoming.output() + self.b.variable()
	
	def get_variable(self):
		return self.b.variable()
	
	def set_trainable(self, status):
		self.b.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.b.set_l1_regularize(status)    
		
	def set_l2_regularize(self, status):
		self.b.set_l2_regularize(status)    
	
	def is_trainable(self):
		return self.b.is_trainable()
		
	def is_l1_regularize(self):
		return self.b.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.b.is_l2_regularize()  


#---------------------------------------------------------------------------------
# useful functions
#---------------------------------------------------------------------------------

		
def activation(z, function='relu', **kwargs):
	if function == 'relu':
		output = tf.nn.relu(z, **kwargs)

	elif function == 'linear':
		output = tf.mult(z, 1, **kwargs)

	elif function == 'sigmoid':
		output = tf.nn.sigmoid(z, **kwargs)

	elif function == 'softmax':
		y_out = tf.nn.sigmoid(z)
		expy = tf.exp(y_out)
		sumexpy = tf.reduce_sum(expy)
		output = tf.div(expy, sumexpy, **kwargs)

	elif function == 'elu':
		output = tf.nn.elu(z, **kwargs)

	elif function == 'softplus':
		output = tf.nn.softplus(z, **kwargs)

	elif function == 'tanh':
		output = tf.nn.tanh(z, **kwargs)

	return output
