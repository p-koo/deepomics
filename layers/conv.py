import tensorflow as tf
from .base import BaseLayer
from ..utils import Variable
from .. import init


__all__ = [
	"Conv1DLayer"
]


class Conv1DLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, num_filters, W=[],
				  strides=[], padding=[], **kwargs):

		self.incoming = incoming
		self.filter_size = filter_size
		self.num_filters = num_filters
		
		dim = incoming.output().get_shape()[3].value
		shape = [filter_size, 1, dim, num_filters]
		self.shape = shape

		if not W:
			self.W = Variable(var=init.HeNormal(), shape=shape, **kwargs)
		else:
			self.W = Variable(var=W, shape=shape, **kwargs)
			
		self.strides = strides
		if not strides:
			self.strides = [1, 1, 1, 1]

		self.padding = padding
		if not padding:
			self.padding = 'VALID'
		
		
	def output(self):
		return tf.nn.conv2d( input=self.incoming.output(), 
							filter=self.W.variable(), 
							strides=self.strides, 
						   	padding=self.padding)
	
	
	def get_variable(self):
		return self.W.variable()
	
	def set_trainable(self, status):
		self.W.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.W.set_l1_regularize(status)    
		
	def set_l2_regularize(self, status):
		self.W.set_l2_regularize(status)    
		
	def is_trainable(self):
		return self.W.is_trainable()
		
	def is_l1_regularize(self):
		return self.W.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.W.is_l2_regularize()  
		
 