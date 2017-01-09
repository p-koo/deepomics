import tensorflow as tf
from .base import BaseLayer

		
__all__ = [
	"MaxPool1DLayer",
]


class MaxPool1DLayer(BaseLayer):
	def __init__(self, incoming, pool_size, strides=[], padding=[], **kwargs):
		
		self.incoming = incoming
		self.pool_size = [1, pool_size, 1, 1]

		if not strides:
			strides = pool_size
		self.strides = [1, strides, 1, 1]
		
		self.padding = padding
		if not self.padding:
			self.padding = 'SAME'
		
	def output(self):
		return tf.nn.max_pool(self.incoming.output(), 
							ksize=self.pool_size, 
							strides=self.strides, 
							padding=self.padding)
	
		