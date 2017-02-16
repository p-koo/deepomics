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

		
		if not W:
			self.W = Variable(var=init.HeUniform(), shape=shape, **kwargs)
		else:
			self.W = Variable(var=W, shape=shape, **kwargs)
			
		if b is None:
			self.b = []
		else:
			if not b:
				self.b = Variable(var=init.Constant(0.05), shape=[num_units], **kwargs)
			else:
				self.b = Variable(var=b, shape=[num_units], **kwargs)
			
		self.incoming_shape = incoming.get_output_shape()
		
		self.output = tf.matmul(incoming.get_output(), self.W.get_variable())
		if self.b:
			self.output += self.b.get_variable()
			
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		if self.b:
			return [self.W, self.b]
		else:
			return self.W
	
	def set_trainable(self, status):
		self.W.set_trainable(status)
		if self.b:
			self.b.set_trainable(status)
			
	def set_l1_regularize(self, status):
		self.W.set_l1_regularize(status)    
		if self.b:
			self.b.set_l1_regularize(status)
		
	def set_l2_regularize(self, status):
		self.W.set_l2_regularize(status)    
		if self.b:
			self.b.set_l2_regularize(status)
	
	def is_trainable(self):
		return self.W.is_trainable()
		
	def is_l1_regularize(self):
		return self.W.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.W.is_l2_regularize()  
	