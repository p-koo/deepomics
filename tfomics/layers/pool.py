import tensorflow as tf
from .base import BaseLayer

		
__all__ = [
	"MaxPool1DLayer",
	"MaxPool2DLayer"
]

class MaxPool1DLayer(BaseLayer):
	def __init__(self, incoming, pool_size, strides=[], padding=[], **kwargs):
		
		self.pool_size = [1, pool_size, 1, 1]

		if not strides:
			strides = pool_size
		self.strides = [1, strides, 1, 1]
		
		self.padding = padding
		if not self.padding:
			self.padding = 'SAME'
		
		self.incoming_shape = incoming.get_output_shape()
		
		self.output = tf.nn.max_pool(incoming.get_output(), 
									ksize=self.pool_size, 
									strides=self.strides, 
									padding=self.padding, 
									**kwargs)
		
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
		
		
class MaxPool2DLayer(BaseLayer):
	def __init__(self, incoming, pool_size, strides=[], padding=[], **kwargs):
		if not isinstance(pool_size, (list, tuple)):
			self.pool_size = [1, pool_size, pool_size, 1]
		else:
			self.pool_size = [1, pool_size[0], pool_size[1], 1]

		if not strides:		
			self.strides = self.pool_size
		else:
			if not isinstance(strides, (list, tuple)):
				self.strides = [1, strides, strides, 1]
			else:
				self.strides = [1, strides[0], strides[1], 1]
		
		self.padding = padding
		if not self.padding:
			self.padding = 'VALID'
		
		self.incoming_shape = incoming.get_output_shape()
		
		self.output = tf.nn.max_pool(incoming.get_output(), 
									ksize=self.pool_size, 
									strides=self.strides, 
									padding=self.padding, 
									**kwargs)
		
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
		
		