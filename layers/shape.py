import tensorflow as tf
from .base import BaseLayer


__all__ = [
	"ReshapeLayer"
]

class ReshapeLayer(BaseLayer):
	def __init__(self, incoming, shape=[], **kwargs):
		self.incoming = incoming
		self.shape = shape
		if not self.shape:
			input_dim = 1
			for dim in incoming.output().get_shape():
				if dim.value:
					input_dim *= dim.value
			self.shape = [-1, input_dim]
	
	def output(self):
		return tf.reshape(self.incoming.output(), self.shape)

	
