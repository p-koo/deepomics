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

		if len(incoming.output().get_shape()) > 2:
			incoming = ReshapeLayer(incoming)
						
		num_inputs = incoming.output().get_shape()[1].value
		self.shape = [num_inputs, num_units]
		
		self.incoming = incoming
		
		if not W:
			self.W = Variable(var=init.HeNormal(), shape=self.shape, **kwargs)
		else:
			self.W = Variable(var=W, shape=self.shape, **kwargs)
			
		if not b:
			self.b = Variable(var=init.Constant(0.05), shape=[num_units], **kwargs)
		else:
			self.b = Variable(var=b, shape=[num_units], **kwargs)
			
		
	def output(self):

		val = tf.matmul(self.incoming.output(), self.W.variable())
		if self.b:
			val+= self.b.variable()
		return val
	
	def output_shape(self):
		return self.shape
		
	def get_variable(self):
		if self.b:
			return [self.W.variable(), self.b.variable()]
		else:
			return self.W.variable()
	
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
	