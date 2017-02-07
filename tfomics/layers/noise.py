import tensorflow as tf
from .base import BaseLayer

		
__all__ = [
	"DropoutLayer",
]


class DropoutLayer(BaseLayer):
	def __init__(self, incoming, keep_prob=0.5, **kwargs):
				
		self.incoming_shape = incoming.get_output_shape()
		
		self.output = tf.nn.dropout(incoming.get_output(), keep_prob=keep_prob)
		
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
		
	