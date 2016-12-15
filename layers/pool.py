import tensorflow as tf
from .base import BaseLayer

		
__all__ = [
	"MaxPool1DLayer",
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
		
		