import tensorflow as tf
from .base import BaseLayer

		
__all__ = [
	"DropoutLayer",
]


class DropoutLayer(BaseLayer):
	def __init__(self, incoming, keep_prob, is_training, **kwargs):
				
		self.incoming = incoming
		self.keep_prob = keep_prob
		self.is_training = is_training

	def output(self):

		keep_prob = tf.select(self.is_training, self.keep_prob, 1)
		return tf.nn.dropout(self.incoming.output(), keep_prob=keep_prob)
	