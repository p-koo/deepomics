from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .base import BaseLayer
from ..utils import Variable
from .. import init

__all__ = [
	"Conv1DLayer", 
	"Conv2DLayer",
	"TransposeConv1DLayer",
	"TransposeConv2DLayer"
]


class Conv1DLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, num_filters, W=[], b=None,
				  strides=[], padding=[], reverse=False, **kwargs):

		self.filter_size = filter_size
		self.num_filters = num_filters
		
		dim = incoming.get_output_shape()[3].value
		shape = [filter_size, 1, dim, num_filters]
		self.shape = shape

		if not W:
			self.W = Variable(var=init.HeUniform(), shape=shape)
		else:
			self.W = Variable(var=W, shape=shape)

		if reverse:
			W_reverse = tf.reverse(self.W.get_variable(), axis=[0, 2])
			W = tf.concat([self.W.get_variable(), W_reverse], axis=3)
		else:
			W = self.W.get_variable()

		if not strides:
			self.strides = [1, 1, 1, 1]
		else:
			self.strides = [1, strides, 1, 1]

		self.padding = padding
		if not padding:
			self.padding = 'VALID'
			
		# input data shape
		self.incoming_shape = incoming.get_output_shape()
		
		# output of convolution
		self.output = tf.nn.conv2d( input=incoming.get_output(), 
									filter=W, 
									strides=self.strides, 
									padding=self.padding)

		if b is None:
			self.b = []
		else:
			if not b:
				self.b = Variable(var=init.Constant(0.05, kwargs), shape=[num_units])
			else:
				self.b = Variable(var=b, shape=[num_units], **kwargs)
			self.output = tf.nn.bias_add(self.output,self.b.get_variable())

		# shape of the output
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return self.W
	
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
		


class Conv2DLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, num_filters, W=[],
				  strides=[], padding=[], **kwargs):

		self.filter_size = filter_size
		self.num_filters = num_filters
		
		dim = incoming.get_output_shape()[3].value

		if not isinstance(filter_size, (list, tuple)):
			self.shape = [filter_size, filter_size, dim, num_filters]
		else:
			self.shape = [filter_size[0], filter_size[1], dim, num_filters]

		if not W:
			self.W = Variable(var=init.HeUniform(**kwargs), shape=self.shape, **kwargs)
		else:
			self.W = Variable(var=W, shape=self.shape, **kwargs)
			

		if not strides:		
			self.strides = [1, 1, 1, 1]
		else:
			if not isinstance(strides, (list, tuple)):
				self.strides = [1, strides, strides, 1]
			else:
				self.strides = [1, strides[0], strides[1], 1]

		self.padding = padding
		if not padding:
			self.padding = 'VALID'
			
		# input data shape
		self.incoming_shape = incoming.get_output_shape()
		
		# output of convolution
		self.output = tf.nn.conv2d( input=incoming.get_output(), 
									filter=self.W.get_variable(), 
									strides=self.strides, 
									padding=self.padding)
		# shape of the output
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return self.W
	
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
		



#---------------------------------------------------------------------------



class TransposeConv1DLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, num_filters, output_shape, W=[],
				  strides=[], padding=[], **kwargs):

		self.filter_size = filter_size
		self.num_filters = num_filters
		
		dim = incoming.get_output_shape()[3].value
		shape = [filter_size, 1, num_filters, dim]
		self.shape = shape

		if not W:
			self.W = Variable(var=init.HeUniform(**kwargs), shape=shape, **kwargs)
		else:
			self.W = Variable(var=W, shape=shape, **kwargs)
			
		if not strides:		
			self.strides = [1, 1, 1, 1]
		else:
			if not isinstance(strides, (list, tuple)):
				self.strides = [1, strides, 1, 1]
			else:
				self.strides = strides

		self.padding = padding
		if not padding:
			self.padding = 'VALID'
			
		# input data shape
		self.incoming_shape = incoming.get_output_shape()

		output_shape = (tf.shape(incoming.get_output())[0], ) + tuple(output_shape[1:])
		output_shape = tf.stack(list(output_shape))
		self.output_shape = output_shape
		
		# output of convolution
		self.output = tf.nn.conv2d_transpose( input=incoming.get_output(), 
									filter=self.W.get_variable(), 
									output_shape=self.output_shape,
									strides=self.strides, 
									padding=self.padding)
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return self.W
	
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


class TransposeConv2DLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, num_filters, output_shape, W=[],
				  strides=[], padding=[], **kwargs):

		self.filter_size = filter_size
		self.num_filters = num_filters
		
		dim = incoming.get_output_shape()[3].value

		if not isinstance(filter_size, (list, tuple)):
			self.shape = [filter_size, filter_size, num_filters, dim]
		else:
			self.shape = [filter_size[0], filter_size[1], num_filters, dim]

		if not W:
			self.W = Variable(var=init.HeUniform(**kwargs), shape=self.shape, **kwargs)
		else:
			self.W = Variable(var=W, shape=self.shape, **kwargs)
			
		output_shape = (tf.shape(incoming.get_output())[0], ) + tuple(output_shape[1:])
		output_shape = tf.stack(list(output_shape))
		self.output_shape = output_shape

		if not strides:		
			self.strides = [1, 1, 1, 1]
		else:
			if not isinstance(strides, (list, tuple)):
				self.strides = [1, strides, strides, 1]
			else:
				self.strides = [1, strides[0], strides[1], 1]

		self.padding = padding
		if not padding:
			self.padding = 'VALID'
			
		# input data shape
		self.incoming_shape = incoming.get_output_shape()
		
		# output of convolution
		self.output = tf.nn.conv2d_transpose(input=incoming.get_output(), 
											filter=self.W.get_variable(), 
											output_shape=self.output_shape,
											strides=self.strides, 
											padding=self.padding)
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return self.W
	
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
		


