from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .base import BaseLayer
from ..utils import Variable
from .shape import ReshapeLayer
from .. import init


__all__ = [
	"DenseLayer",
]

	
class DenseLayer(BaseLayer):
	"""Fully-connected layer"""

	def __init__(self, incoming, num_units, W=[], b=[], **kwargs):

		self.num_units = num_units
		
		if len(incoming.get_output_shape()) > 2:
			incoming = ReshapeLayer(incoming)
			
		num_inputs = incoming.get_output_shape()[1].value
		shape = [num_inputs, num_units]
		self.shape = shape
		self.incoming_shape = incoming.get_output_shape()
		
		if not W:
			self.W_flat = Variable(var=init.HeUniform(), shape=shape)
		else:
			self.W_flat = Variable(var=W, shape=shape)
		self.W = tf.reshape(self.W_flat.get_variable(), shape=shape)

		if b is None:
			self.b = []
		else:
			if not b:
				self.b = Variable(var=init.Constant(0.05), shape=[num_units], **kwargs)
			else:
				self.b = Variable(var=b, shape=[num_units], **kwargs)
			
		self.output = tf.matmul(incoming.get_output(), self.W)

		if self.b:
			self.output = tf.nn.bias_add(self.output, self.b.get_variable())
			
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self, shape=False):
		if shape:
			if self.b:
				return [self.W, self.b.get_variable()]
			else:
				return self.W
		else:
			if self.b:
				return [self.W_flat.get_variable(), self.b.get_variable()]
			else:
				return self.W_flat.get_variable()

	def set_trainable(self, status):
		self.W_flat.set_trainable(status)
		if self.b:
			self.b.set_trainable(status)
			
	def set_l1_regularize(self, status):
		self.W_flat.set_l1_regularize(status)    
		if self.b:
			self.b.set_l1_regularize(status)
		
	def set_l2_regularize(self, status):
		self.W_flat.set_l2_regularize(status)    
		if self.b:
			self.b.set_l2_regularize(status)
	
	def is_trainable(self):
		return self.W_flat.is_trainable()
		
	def is_l1_regularize(self):
		return self.W_flat.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.W_flat.is_l2_regularize()  
	