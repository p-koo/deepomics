import tensorflow as tf
from .base import BaseLayer
from ..utils import Variable
from .. import init

__all__ = [
	"BatchNormLayer"
]


class BatchNormLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, is_training, **kwargs):

		# input data shape
		self.incoming = incoming
		shape = incoming.output().get_shape()[-1].value

		self.gamma = Variable(var=init.Constant(value=1.), shape=[shape], **kwargs)
		self.beta = Variable(var=init.Constant(value=0.), shape=[shape], **kwargs)

		self.pop_mean = tf.Variable(tf.zeros(shape), trainable=False)
		self.pop_var = tf.Variable(tf.ones(shape), trainable=False)

		self.epsilon = 1e-08
		if 'epsilon' in kwargs.keys():
			self.epsilon = kwargs['epsilon']
		self.decay = 0.999
		if 'decay' in kwargs.keys():
			self.epsilon = kwargs['decay']

		self.is_training = is_training

	
	def output(self):
		return tf.select(self.is_training, self._batch_output(), self._pop_output())

	def _batch_output(self):
		batch_mean, batch_var = tf.nn.moments(self.incoming.output(),[0,1,2])
		train_mean = tf.assign(self.pop_mean, self.pop_mean*self.decay + batch_mean*(1 - self.decay))
		train_var = tf.assign(self.pop_var, self.pop_var*self.decay + batch_var*(1 - self.decay))
		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(self.incoming.output(), batch_mean, batch_var, 
		                                 self.beta.variable(), self.gamma.variable(), self.epsilon)
	
	def _pop_output(self):
		return tf.nn.batch_normalization(self.incoming.output(), self.pop_mean, self.pop_var, 
				                                 self.beta.variable(), self.gamma.variable(), self.epsilon)			
	
	def get_variable(self):
		return [self.gamma.variable(), self.beta.variable()]

	def set_trainable(self, status):
		self.gamma.set_trainable(status)
		self.beta.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.gamma.set_l1_regularize(status)  
		self.beta.set_l1_regularize(status)   
		
	def set_l2_regularize(self, status):
		self.gamma.set_l2_regularize(status)  
		self.beta.set_l2_regularize(status)     

	def is_trainable(self):
		return self.gamma.is_trainable()
		
	def is_l1_regularize(self):
		return self.gamma.is_l1_regularize()  
		
	def is_l2_regularize(self):
		return self.gamma.is_l2_regularize() 
